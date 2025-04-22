import streamlit as st
import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
import logging
import json
import re
import networkx as nx
import plotly.graph_objects as go

# Set page config as the first Streamlit command
st.set_page_config(page_title="Causality Tagging App", layout="wide")

# Initialize session state
if "output_data" not in st.session_state:
    st.session_state.output_data = None

# Load environment variables
load_dotenv()

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Neo4j graph
NEO4j_URI = 'bolt://localhost:7687'
NEO4j_USER = 'neo4j'
NEO4j_PASSWORD = os.environ.get('pass')
graph = Neo4jGraph(url=NEO4j_URI, username=NEO4j_USER, password=NEO4j_PASSWORD)

# LLM selection
llm_options = {
    "qwen-qwq-32b": "qwen-qwq-32b",
    "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
    "llama-3.3-70b-specdec": "llama-3.3-70b-specdec",
    "deepseek-r1-distill-llama-70b": "deepseek-r1-distill-llama-70b",
    "deepseek-r1-distill-qwen-32b": "deepseek-r1-distill-qwen-32b"
}

selected_llm = st.selectbox("Select LLM:", list(llm_options.keys()))

# Initialize LLM based on selection
if selected_llm == "qwen-qwq-32b":
    llm = ChatGroq(
        groq_api_key=os.environ.get('GROQ_API'),
        model_name='qwen-qwq-32b'
    )
elif selected_llm == "llama-3.3-70b-versatile":
    llm = ChatGroq(
        groq_api_key=os.environ.get('GROQ_API'),
        model_name='llama-3.3-70b-versatile'
    )
elif selected_llm == "llama-3.3-70b-specdec":
    llm = ChatGroq(
        groq_api_key=os.environ.get('GROQ_API'),
        model_name='llama-3.3-70b-specdec'
    )
elif selected_llm == "deepseek-r1-distill-llama-70b":
    llm = ChatGroq(
        groq_api_key=os.environ.get('GROQ_API'),
        model_name='deepseek-r1-distill-llama-70b'
    )
elif selected_llm == "deepseek-r1-distill-qwen-32b":
    llm = ChatGroq(
        groq_api_key=os.environ.get('GROQ_API'),
        model_name='deepseek-r1-distill-qwen-32b'
    )

def retrieve_similar_events(query_text, graph, embeddings, top_k=5, embedding_weight=0.6, structure_weight=0.4, similarity_threshold=0.5):
    """
    Retrieve similar events using a hybrid approach with optimized query performance.
    """
    query_embedding = embeddings.embed_query(query_text)
    
    cypher_query = """
    MATCH (e:Event)
    WHERE e.embedding IS NOT NULL AND e.text IS NOT NULL
    
    // Efficient connection counting using COUNT
    OPTIONAL MATCH (e)-[:RESULTS_IN]->(effect:Effect)
    WITH e, COUNT(effect) AS effect_count
    
    OPTIONAL MATCH (e)<-[:CAUSES]-(cause:Cause)
    WITH e, effect_count, COUNT(cause) AS cause_count
    
    OPTIONAL MATCH (e)-[:HAS_TRIGGER]->(trigger:Trigger)
    WITH e, effect_count, cause_count, COUNT(trigger) AS trigger_count
    
    // Calculate structural score (maintain original binary approach)
    WITH e, effect_count, cause_count, trigger_count,
         CASE 
             WHEN effect_count + cause_count + trigger_count = 0 
             THEN 0.0 
             ELSE 1.0 
         END AS structural_score
    
    // Calculate similarity and hybrid score
    WITH e, effect_count, cause_count, trigger_count, structural_score,
         gds.similarity.cosine(e.embedding, $query_embedding) AS embedding_similarity,
         ($embedding_weight * gds.similarity.cosine(e.embedding, $query_embedding) + $structure_weight * structural_score) AS hybrid_score
    
    WHERE hybrid_score >= $similarity_threshold
    
    // Late text collection for final candidates
    CALL {
        WITH e
        OPTIONAL MATCH (e)-[:RESULTS_IN]->(effect:Effect)
        RETURN COLLECT(effect.text) AS effect_texts
    }
    CALL {
        WITH e
        OPTIONAL MATCH (e)<-[:CAUSES]-(cause:Cause)
        RETURN COLLECT(cause.text) AS cause_texts
    }
    CALL {
        WITH e
        OPTIONAL MATCH (e)-[:HAS_TRIGGER]->(trigger:Trigger)
        RETURN COLLECT(trigger.text) AS trigger_texts
    }
    
    RETURN e.text AS text,
           e.id AS event_id,
           hybrid_score,
           embedding_similarity,
           structural_score,
           effect_count,
           cause_count,
           trigger_count,
           effect_texts,
           cause_texts,
           trigger_texts
    ORDER BY hybrid_score DESC
    LIMIT $top_k
    """
    
    try:
        logging.getLogger("neo4j").setLevel(logging.ERROR)
        results = graph.query(
            cypher_query,
            {
                "query_embedding": query_embedding,
                "embedding_weight": float(embedding_weight),
                "structure_weight": float(structure_weight),
                "similarity_threshold": float(similarity_threshold),
                "top_k": int(top_k)
            }
        )
        
        similar_events = []
        for record in results:
            if record['text']:
                similar_events.append({
                    'text': record['text'],
                    'event_id': record.get('event_id', 'N/A'),
                    'hybrid_score': round(float(record['hybrid_score']), 3),
                    'embedding_similarity': round(float(record['embedding_similarity']), 3),
                    'structural_score': round(float(record['structural_score']), 3),
                    'connection_counts': {
                        'effects': int(record['effect_count']),
                        'causes': int(record['cause_count']),
                        'triggers': int(record['trigger_count'])
                    },
                    'connected_nodes': {
                        'causes': [str(cause) for cause in record['cause_texts'] if cause],
                        'effects': [str(effect) for effect in record['effect_texts'] if effect],
                        'triggers': [str(trigger) for trigger in record['trigger_texts'] if trigger]
                    }
                })
        
        return similar_events
    
    except Exception as e:
        print(f"Error during query execution: {str(e)}")
        return []

def generate(llm, input, graph, embeddings):
    prompt="""
    <task>
        You are an advanced language model tasked with identifying and tagging explicit cause-and-effect relationships in complex sentences. 
        <rule>Do not modify the original text when adding tags.</rule>
    </task>

    <VERY IMPORTANT RULE>Do not give output unless all tags are present in sentence (All tags meaning cause, effect, and trigger should be present). If you miss this in the end the answer will be considered wrong</VERY IMPORTANT RULE>

    <steps>
        <step number="1">
            <title>Causality Determination</title>
            <description>
                <point>Analyze the input sentence to determine if it contains an explicit causal relationship.</point>
                <point>If the sentence is <b>not causal</b>, output <b>NoTag</b> and set the label to <b>0</b>. <rule>Do not perform any tagging.</rule></point>
                <point>If the sentence is causal, proceed only if <b>all three elements</b>—cause, trigger, and effect—are present.</point>
                <point>If any element is missing, output <b>NoTag</b> and set the label to <b>0</b>.</point>
            </description>
        </step>

        <step number="2">
            <title>Tagging Instructions</title>
            <tags>
                <tag>
                    <name>Cause</name>
                    <format>&lt;cause&gt;...&lt;/cause&gt;</format>
                    <definition>The event, action, or condition that leads to an outcome.</definition>
                    <question>What caused this?</question>
                </tag>
                <tag>
                    <name>Trigger</name>
                    <format>&lt;trigger&gt;...&lt;/trigger&gt;</format>
                    <definition>The word or phrase that explicitly indicates causality (e.g., because, due to).</definition>
                </tag>
                <tag>
                    <name>Effect</name>
                    <format>&lt;effect&gt;...&lt;/effect&gt;</format>
                    <definition>The outcome or result of the cause.</definition>
                    <question>What is the result of this?</question>
                </tag>
            </tags>
        </step>

        <step number="3">
            <title>Causality Tests</title>
            <description>
                <point>The effect should answer "Why?" (i.e., it must clearly result from the cause).</point>
                <point>The cause must precede the effect in time.</point>
                <point>The effect should not occur without the cause.</point>
                <point>The cause and effect cannot be swapped without changing the meaning.</point>
                <point>The sentence can be rephrased as “X causes Y” or “Due to X, Y.”</point>
            </description>
        </step>

        <step number="4">
            <title>Additional Guidelines</title>
            <guidelines>
                <point>The sentence must have explicit cause-effect wording.</point>
                <point>If there are multiple causes or effects, tag each instance separately.</point>
                <point>Do not alter the original sentence structure when adding tags.</point>
                <point>The final output must include all three tags: <b>&lt;cause&gt;</b>, <b>&lt;trigger&gt;</b>, and <b>&lt;effect&gt;</b>.</point>
                <point>If any tag is missing, consider the sentence non-causal.</point>
                <point>If no explicit causal relationship is identified, output <b>NoTag</b>.</point>
            </guidelines>
        </step>

        <step number="5">
            <title>Output Format</title>
            <output>
                <format>Return a JSON object with two keys: "tagged" and "label".</format>
                <key name="tagged">Contains the original sentence with all tags added.</key>
                <key name="label">Set to <b>1</b> if a causal relationship is successfully tagged; otherwise, <b>0</b>.</key>
            </output>
        </step>
    </steps>

    <examples>
        <example>
            <input>Biswal, who had travelled to the city from Washington to speak at the Indian Consulate General's Media-India Lecture Series yesterday, said not only were there a large number of Indian victims in the attack but there were Americans also who lost their lives.</input>
            <tagged>Biswal, who had <trigger>travelled to the city from Washington</trigger> to <cause>speak at the Indian Consulate General's Media-India Lecture Series</cause> yesterday, said <effect>not only were there a large number of Indian victims in the attack</effect> but <effect>there were Americans also who lost their lives</effect>.</tagged>
            <label>1</label>
        </example>

        <more examples>
        {similar_examples}
        </more examples>

        <example>
            <input>The African nationals had sustained minor injuries in the attack that took place in South Delhi's Mehrauli area on Friday.</input>
            <tagged>NoTag</tagged>
            <label>0</label>
        </example>
        
        <example>
            <input>On Sunday , Swaraj spoke to Rajnath Singh and Delhi Lieutenant Governor Najeeb Jung about the attacks and said she was assured that the culprits would soon be arrested .</input>
            <tagged>NoTag</tagged>
            <label>0</label>
        </example>

        <example>
            <input>We will continue our strike till we get an assurance from the Government .</input>
            <tagged>NoTag</tagged>
            <label>0</label>
        </example>

        <example>
            <input>There is no information about the involvement of the NDFB(S) underground group in the blast , he added .</input>
            <tagged>NoTag</tagged>
            <label>0</label>
        </example>

    </examples>

    <qa>
        <requirement>For sentences that are tagged as causal, include a brief description explaining the cause-and-effect relationship.</requirement>
        <example>
            <sentence>&lt;effect>The participants raised slogans&lt;/effect> &lt;trigger>demanding&lt;/trigger> &lt;cause>stern action to check atrocities against women in the name of moral policing&lt;/cause></sentence>
            <explanation>Why did they raise slogans? To demand action against atrocities.</explanation>
            <label>1</label>
        </example>
    </qa>

    <final_instruction>
        Now, process the following input text:
        <input>{input_text}</input>
        <note>Reconsider your decisions step by step internally (chain-of-thought) but do not output your reasoning.</note>
        <IMPORTANT>Do not give output unless all tags are present in sentence (All tags meaning cause, effect, and trigger should be present).</IMPORTANT>
    </final_instruction>
    """
    prompt_template = PromptTemplate(
        input_variables=["similar_examples", "input_text"],
        template=prompt
    )

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    )

    similar_events = retrieve_similar_events(
        input,
        graph,
        embeddings,
        top_k=20,
        embedding_weight=0.6,
        structure_weight=0.4,
        similarity_threshold=0.5
    )
    
    output = llm_chain.run(
        similar_examples=similar_events,
        input_text=input
    )

    return output

def generate_safe_headline(llm, tagged_text):
    prompt = """
    <task>
        You are an advanced language model tasked with generating safe headlines from tagged text.
        <rule>Remove any potentially harmful or sensitive words from the headline.</rule>
    </task>

    <steps>
        <step number="1">
            <title>Identify Sensitive Content</title>
            <description>
                <point>Analyze the tagged text to identify any words or phrases that might be considered sensitive or harmful.</point>
                <point>Look for words related to violence, conflict, hate speech, or other potentially harmful topics.</point>
                <point>Do not remove any dates in the headline, as they are very crucial.</point>
            </description>
        </step>

        <step number="2">
            <title>Generate Safe Headline</title>
            <description>
                <point>Create a headline that conveys the main information without including any sensitive content.</point>
                <point>Maintain the core meaning of the original text while removing potentially harmful elements.</point>
            </description>
        </step>

        <step number="3">
            <title>Generate Analysis</title>
            <description>
                <point>Provide a brief analysis of why certain words were removed or modified.</point>
                <point>Explain the reasoning behind the changes made to ensure safety.</point>
            </description>
        </step>

        <step number="4">
            <title>Output Format</title>
            <output>
                Output the safe headline first, followed by the analysis. Format the output as:
                <headline> [safe headline] </headline>
                <analysis> [analysis text] </analysis>
            </output>
        </step>
    </steps>

    <final_instruction>
        Now, process the following tagged text and generate a safe headline and analysis:
        <input>{tagged_text}</input>
    </final_instruction>
    """
    prompt_template = PromptTemplate(
        input_variables=["tagged_text"],
        template=prompt
    )

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    )

    output = llm_chain.run(tagged_text=tagged_text)
    
    # Log the raw output for debugging
    print(f"Safe headline raw output: {output}")

    return output

def create_interactive_graph(similar_events):
    G = nx.DiGraph()
    
    for event in similar_events:
        event_node = f"Event {event['event_id']}"
        G.add_node(event_node, text=event['text'], hybrid_score=event['hybrid_score'])
        
        # Add connections to causes
        for cause in event['connected_nodes']['causes']:
            cause_node = f"Cause: {cause}"
            G.add_node(cause_node, type='cause')
            G.add_edge(event_node, cause_node, relation='HAS_CAUSE')
        
        # Add connections to effects
        for effect in event['connected_nodes']['effects']:
            effect_node = f"Effect: {effect}"
            G.add_node(effect_node, type='effect')
            G.add_edge(event_node, effect_node, relation='HAS_EFFECT')
        
        # Add connections to triggers
        for trigger in event['connected_nodes']['triggers']:
            trigger_node = f"Trigger: {trigger}"
            G.add_node(trigger_node, type='trigger')
            G.add_edge(event_node, trigger_node, relation='HAS_TRIGGER')
    
    return G

def create_processed_text_graph(tagged_text):
    G = nx.DiGraph()
    
    # Add event node
    event_node = "Processed Event"
    G.add_node(event_node, text=tagged_text, type='event')
    
    # Extract causes, effects, and triggers
    causes = re.findall(r'<cause>(.*?)</cause>', tagged_text)
    effects = re.findall(r'<effect>(.*?)</effect>', tagged_text)
    triggers = re.findall(r'<trigger>(.*?)</trigger>', tagged_text)
    
    # Add cause nodes and edges
    for cause in causes:
        cause_node = f"Cause: {cause}"
        G.add_node(cause_node, type='cause')
        G.add_edge(event_node, cause_node, relation='HAS_CAUSE')
    
    # Add effect nodes and edges
    for effect in effects:
        effect_node = f"Effect: {effect}"
        G.add_node(effect_node, type='effect')
        G.add_edge(event_node, effect_node, relation='HAS_EFFECT')
    
    # Add trigger nodes and edges
    for trigger in triggers:
        trigger_node = f"Trigger: {trigger}"
        G.add_node(trigger_node, type='trigger')
        G.add_edge(event_node, trigger_node, relation='HAS_TRIGGER')
    
    return G

def draw_interactive_graph(G):
    pos = nx.spring_layout(G)
    
    # Create nodes
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        if 'Event' in node:
            node_colors.append('lightblue')
        elif 'Cause' in node:
            node_colors.append('lightgreen')
        elif 'Effect' in node:
            node_colors.append('lightcoral')
        elif 'Trigger' in node:
            node_colors.append('lightyellow')
    
    # Create edges
    edge_x = []
    edge_y = []
    edge_text = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_text.append(edge[2].get('relation', ''))
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=15,
            color=node_colors,
            line_width=2
        ),
        text=node_text,
        textposition='top center',
        hoverinfo='text',
        textfont=dict(size=10, color='white')
    ))
    
    # Configure layout
    fig.update_layout(
        title=dict(
            text='Event Relationship Graph',
            font=dict(size=16)
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[dict(
            text="",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002 
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        clickmode='event+select'
    )
    
    return fig

# Streamlit app
st.title("Causality Tagging App")

# Legend for node colors
st.markdown("""
<style>
.legend {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}
.legend-item {
    margin: 0 10px;
    display: flex;
    align-items: center;
}
.legend-color {
    width: 15px;
    height: 15px;
    margin-right: 5px;
}
.event-color { background-color: lightblue; }
.cause-color { background-color: lightgreen; }
.effect-color { background-color: lightcoral; }
.trigger-color { background-color: lightyellow; }
</style>
<div class="legend">
    <div class="legend-item">
        <div class="legend-color event-color"></div>
        <span>Event</span>
    </div>
    <div class="legend-item">
        <div class="legend-color cause-color"></div>
        <span>Cause</span>
    </div>
    <div class="legend-item">
        <div class="legend-color effect-color"></div>
        <span>Effect</span>
    </div>
    <div class="legend-item">
        <div class="legend-color trigger-color"></div>
        <span>Trigger</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Input area
input_text = st.text_area("Enter your text here:", height=150)

# Process button
process_button = st.button("Causality Extraction")

# Process text when button is clicked
if process_button:
    if not input_text.strip():
        st.error("Please enter some text to process.")
    else:
        with st.spinner("Processing..."):
            # Generate tagged output
            llm_output = generate(llm, input_text, graph, embeddings)
            match = re.search(r"\{.*\}", llm_output, re.DOTALL)
            if match:
                json_data = match.group(0)
                try:
                    output_data = json.loads(json_data)
                    st.session_state.output_data = output_data
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON.")
            else:
                print(f"Error: No JSON found")
            
            # Display results
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.subheader("Tagged Sentence")
                if output_data["label"] == 1:
                    st.markdown(f"<span style='color: green'>Causal Relationship Found</span>", unsafe_allow_html=True)
                    st.write(output_data["tagged"])
                    
                    # Extract and display cause, effect, trigger
                    st.subheader("Causality Elements")
                    cause = re.search(r'<cause>(.*?)</cause>', output_data["tagged"])
                    effect = re.search(r'<effect>(.*?)</effect>', output_data["tagged"])
                    trigger = re.search(r'<trigger>(.*?)</trigger>', output_data["tagged"])
                    
                    if cause:
                        st.write(f"**Cause:** {cause.group(1)}")
                    if effect:
                        st.write(f"**Effect:** {effect.group(1)}")
                    if trigger:
                        st.write(f"**Trigger:** {trigger.group(1)}")
                else:
                    st.markdown(f"<span style='color: red'>No Causal Relationship Found</span>", unsafe_allow_html=True)
                    st.write(output_data["tagged"])
            
            with col2:
                st.subheader("Processed Text Graph")
                if output_data["label"] == 1:
                    G = create_processed_text_graph(output_data["tagged"])
                    fig = draw_interactive_graph(G)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No causal relationship to visualize.")
            
            with col3:
                st.subheader("Similar Events")
                similar_events = retrieve_similar_events(
                    input_text,
                    graph,
                    embeddings,
                    top_k=5,
                    embedding_weight=0.6,
                    structure_weight=0.4,
                    similarity_threshold=0.5
                )
                
                if similar_events:
                    # Create and display interactive graph
                    G = create_interactive_graph(similar_events)
                    fig = draw_interactive_graph(G)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display event details
                    for event in similar_events:
                        with st.expander(f"Event ID: {event['event_id']} (Hybrid Score: {event['hybrid_score']})"):
                            st.write(f"**Text:** {event['text']}")
                            st.write(f"**Connection Counts:**")
                            st.write(f"- Causes: {event['connection_counts']['causes']}")
                            st.write(f"- Effects: {event['connection_counts']['effects']}")
                            st.write(f"- Triggers: {event['connection_counts']['triggers']}")
                            st.write("---")
                else:
                    st.write("No similar events found.")

# Generate safe headline button
generate_headline_button = st.button("Generate Safe Headline")

if generate_headline_button:
    if st.session_state.output_data is None:
        st.error("Please process text first before generating a safe headline.")
    else:
        with st.spinner("Generating safe headline..."):
            try:
                llm_safe = ChatGroq(
                    groq_api_key=os.environ.get('GROQ_API'),
                    model_name='deepseek-r1-distill-qwen-32b'
                )
                safe_headline = generate_safe_headline(llm_safe, st.session_state.output_data["tagged"])

                headline_match = re.search(r'<headline>(.*?)</headline>', safe_headline, re.DOTALL)
                analysis_match = re.search(r'<analysis>(.*?)</analysis>', safe_headline, re.DOTALL)
                
                if not headline_match or not analysis_match:
                    raise ValueError("LLM output format is incorrect. Expected <headline> and <analysis> tags")
                
                headline = headline_match.group(1).strip()
                analysis = analysis_match.group(1).strip()
                
                # Display the safe headline first, bold and larger
                st.header(f"{headline}")
                st.write(f"{analysis}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")