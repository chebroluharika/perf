import json
import os
import sys
from datetime import datetime

import streamlit as st
import yaml
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

sys.path.append("..")

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import \
    WatsonxLLM
from ibm_watson_machine_learning.metanames import \
    GenTextParamsMetaNames as GenParams
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_community.llms import Ollama

from backend.auth.auth import authenticate
from backend.ceph_operations import Ceph
from backend.config import USE_CASES
from backend.perfomance import Performance

load_dotenv("auth.env")  # Ensure this loads the environment variables

api_key = os.getenv("WATSONX_API_KEY")

if not api_key:
    raise ValueError(
        "❌ Missing Watsonx API Key! Set WATSONX_API_KEY in your .env file."
    )

print(f"✅ Loaded Watsonx API Key: {api_key[:1]}... (masked for security)")

# Set environment variables
os.environ["LANGSMITH_TRACING"] = "true"

global ceph_admin
ceph_admin = None

# Streamlit UI Configuration
st.set_page_config(
    page_title="🚀 Ceph Perf AI Agent Bot",
    initial_sidebar_state="collapsed",
    page_icon="🤖",
    layout="wide",
)

# Apply styling
st.markdown(
    """
    <style>
        .column-container {
            height: 100px;
            border: 10px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            background-color: #f9f9f9;
            margin: 5px;
            box-shadow: 4px 2px 10px rgba(0,0,0,0.1);
        }
        .chat-container, .command-container {
            height: 50px;
            overflow-y: auto;
            border: 2px solid #ddd;
            padding: 15px;
            background-color: white;
            border-radius: 8px;
        }
        body {
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Authentication state
if "authenticated_user" not in st.session_state:
    st.session_state.authenticated_user = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ceph_admin" not in st.session_state:
    st.session_state.ceph_admin = None
if "command_result" not in st.session_state:
    st.session_state.command_result = None


def generate_summary(workload, performance_tunings):
    # Create a prompt template for summarizing
    summary_prompt = """
    You are a performance tuning assistant. Given the following performance tuning data for WORKLOAD, generate a summary that emphasizes settings that need attention and their corresponding recommendations. For each tuning setting, you should output a brief summary.
    Performance Tuning Data:
    {performance_tuning_data}

    Please summarize the data, and mention the settings that need changes, along with the corresponding commands.
    """.replace(
        "WORKLOAD", workload.lower()
    )

    # Set up LangChain LLM with a prompt template
    llm = Ollama(model="llama3")

    # Define the prompt with structured data input
    prompt = PromptTemplate(
        input_variables=["performance_tuning_data"], template=summary_prompt
    )

    # Create a sequence to process the input
    chain = prompt | llm  # This creates a RunnableSequence

    # Generate summary using `.invoke()` (new LangChain standard)
    summary_data = chain.invoke({"performance_tuning_data": str(performance_tunings)})

    return summary_data


# Summary of the data
def tabulate_summary(workload, performance_tunings):
    def _summary(tuning_data):
        total_settings = len(tuning_data)
        optimally_configured = sum(1 for item in tuning_data if "✅" in item["Status"])
        needs_attention = total_settings - optimally_configured
        recommended_commands = [
            item["Run Command"] for item in tuning_data if item["Run Command"]
        ]

        # Generating summary text
        summary = (
            f"Summary: Performance tunables for {workload.lower()} use case\n\n"
            f"- **Total Settings:** {total_settings}\n"
            f"- **Optimally Configured:** {optimally_configured}\n"
            f"- **Settings Needing Attention:** {needs_attention}\n"
        )

        if recommended_commands:
            summary += f"\n- **Recommended Configuration Commands:**\n"
            for cmd in recommended_commands:
                summary += f"    - `{cmd}`\n"

        return summary

    formatted_data = []

    # Loop through the data and adjust based on status for more clarity
    for item in performance_tunings:
        formatted_data.append(
            {
                "Setting": item["Setting"],
                "Status": item["Status"],  # Status with icon
                "Recommendation": item["Recommendation"],  # Recommendation with icon
                "Description": item["Description"],  # Description with icon
                "Run Command": (
                    item["Run Command"] if item["Run Command"] else "N/A"
                ),  # Adding N/A if no command
            }
        )

    return _summary(performance_tunings), formatted_data


ceph_ops = Ceph()
perf_ops = Performance()


def get_ceph_status(_):
    if not st.session_state.authenticated_user:
        return "⚠️ Please log in first."
    status = ceph_ops.ceph_status(st.session_state.ceph_admin)
    return status if status else "Failed to retrieve Ceph status."


def recommend_perf_tunables_low_latency_dbs(_):
    key = "Low Latencys databases workload"
    if not st.session_state.authenticated_user:
        return "⚠️ Please log in first."
    status = perf_ops.summarize(
        client=st.session_state.ceph_admin, params=USE_CASES["LOW_LATENCY_DBS_WORKLOAD"]
    )
    tabular_summary = tabulate_summary(key, status)
    general_summary = generate_summary(key, status)

    response = {
        "general_summary": general_summary,
        "tabular_summary": tabular_summary,
    }
    return (
        json.dumps(response)
        if response
        else f"Failed to retrieve perf recommendations tunbales for {key}."
    )


def recommend_perf_tunables_high_throughput(_):
    key = "High throughput workload"
    if not st.session_state.authenticated_user:
        return "⚠️ Please log in first."
    status = perf_ops.summarize(
        client=st.session_state.ceph_admin,
        params=USE_CASES["HIGH_THROUGHPUT_WORKLOADS"],
    )

    tabular_summary = tabulate_summary(key, status)
    general_summary = generate_summary(key, status)

    response = {
        "general_summary": general_summary,
        "tabular_summary": tabular_summary,
    }
    return (
        json.dumps(response)
        if response
        else f"Failed to retrieve perf recommendations tunbales for {key}."
    )


def recommend_perf_tunables_vm_storage(_):
    key = "Vitualization environment workload"
    if not st.session_state.authenticated_user:
        return "⚠️ Please log in first."
    status = perf_ops.summarize(
        client=st.session_state.ceph_admin, params=USE_CASES["VIRTUAL_MACHINE_STORAGE"]
    )

    tabular_summary = tabulate_summary(key, status)
    general_summary = generate_summary(key, status)

    response = {
        "general_summary": general_summary,
        "tabular_summary": tabular_summary,
    }
    return (
        json.dumps(response)
        if response
        else f"Failed to retrieve perf recommendations tunbales for {key}."
    )


def recommend_perf_tunables_big_data(_):
    key = "Big Data environment workload"
    if not st.session_state.authenticated_user:
        return "⚠️ Please log in first."
    status = perf_ops.summarize(
        client=st.session_state.ceph_admin, params=USE_CASES["BIG_DATA_ANALYTICS"]
    )

    tabular_summary = tabulate_summary(key, status)
    general_summary = generate_summary(key, status)

    response = {
        "general_summary": general_summary,
        "tabular_summary": tabular_summary,
    }
    return (
        json.dumps(response)
        if response
        else f"Failed to retrieve perf recommendations tunbales for {key}."
    )


def parse_ceph_status(status_json):
    status = json.loads(status_json) if isinstance(status_json, str) else status_json
    parsed_data = {
        "Health": str(status.get("health", {}).get("status", "Unknown")),
        "FSID": str(status.get("fsid", "N/A")),
        "Monitor Quorum": "\n\r".join(status.get("quorum_names", [])),
        "Total OSDs": str(status.get("osdmap", {}).get("num_osds", 0)),
        "Up OSDs": str(status.get("osdmap", {}).get("num_up_osds", 0)),
        "Pools": str(status.get("pgmap", {}).get("num_pools", 0)),
        "Objects": str(status.get("pgmap", {}).get("num_objects", 0)),
        "Bytes Used": str(status.get("pgmap", {}).get("bytes_used", 0)),
        "Total PGs": str(status.get("pgmap", {}).get("num_pgs", 0)),
    }
    return parsed_data


def parse_available_hosts(yaml_data):
    hosts = yaml.safe_load_all(yaml_data)
    table_data = []
    for host in hosts:
        table_data.append(
            {
                "Hostname": host["hostname"],
                "Address": host["addr"],
                "Status": host["status"],
            }
        )
    return table_data


def parse_live_services(yaml_data):
    services = yaml.safe_load_all(yaml_data)
    service_list = []
    for service in services:
        service_list.append(
            {
                "Service Name": str(service.get("service_name", "N/A")),
                "Running": f"{str(service.get('status', {}).get('running', 0))}/{str(service.get('status', {}).get('size', 0))}",
            }
        )
    return service_list


suggested_questions = [
    "low latency Databases (MySQL, PostgreSQL, MongoDB, etc.)",
    "Virtualization environment(OpenStack, KVM, Proxmox)",
    "High Performance computing Workloads (Media Streaming, Backup Storage)",
    "Big Data Analytics (Hadoop, Spark, Elasticsearch)",
    "Get me Current Ceph status.",
]

# Memory and LLM
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

generate_params = {GenParams.MAX_NEW_TOKENS: 25}
model = Model(
    model_id=os.getenv("MODEL"),  # Specify the model ID
    credentials={
        "apikey": os.getenv("WATSONX_API_KEY"),
        "url": os.getenv("WATSONX_URL"),
    },  # Get API key and URL from environment variables
    params=generate_params,  # Set generation parameters
    project_id=os.getenv("PROJECT_ID"),  # Get project ID from environment variables
)

# Wrap the model with WatsonxLLM to use with LangChain
llm = WatsonxLLM(model=model)

# Define Tools
tools = [
    Tool(
        name="Ceph Status",
        func=get_ceph_status,
        description="Live Ceph-status.",
        return_direct=True,
    ),
    Tool(
        name="Get performance recommendations for low latency databases",
        func=recommend_perf_tunables_low_latency_dbs,
        description="Retrieve perf recommendations tunbales for Low latency Databases.",
        return_direct=True,
    ),
    Tool(
        name="Get performance recommendations for high throughput workloads",
        func=recommend_perf_tunables_high_throughput,
        description="Retrieve perf recommendations tunbales for high throughput workloads.",
        return_direct=True,
    ),
    Tool(
        name="Get performance recommendations for virtualization environment",
        func=recommend_perf_tunables_vm_storage,
        description="Retrieve perf recommendations tunbales for Virtualization products.",
        return_direct=True,
    ),
    Tool(
        name="Get performance recommendations for big data analytics",
        func=recommend_perf_tunables_big_data,
        description="Retrieve perf recommendations tunbales for big data analytics.",
        return_direct=True,
    ),
]

# Memory and LLM
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = Ollama(model="llama3")

# Initialize AI Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)


def is_response_tabular(response):
    """
    Checks if the response contains structured data that can be tabulated.
    """
    try:
        json_data = json.loads(response)
        if "tabular_summary" in json_data:
            return json_data
    except Exception:
        return False


def process_query(query: str):
    """Process query using Agent.

    Args:
        query: User query
    Returns:
        response
    """
    if not st.session_state.authenticated_user:
        return "⚠️ Please log in first."

    with st.status("🤖 Processing your query... Please wait", expanded=True) as status:
        timestamp = f"📅 {datetime.now().strftime('%A, %d %B %Y, %H:%M:%S')}"
        response = agent.run(query)
        st.session_state.chat_history.append((timestamp, query, response))
        status.update(label="✅ Response Ready!", state="complete", expanded=False)

    return response


# Login access
if not st.session_state.authenticated_user:
    st.markdown(
        "<h1 style='text-align: center;'>🚀 Ceph Perf AI Agent</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h4 style='text-align: justify;'>🔑 Provide Ceph Admin Access</h4>",
        unsafe_allow_html=True,
    )
    with st.form("login_form"):
        host = st.text_input("Ceph Admin Host:", key="host IP address")
        username = st.text_input("Username:", key="username")
        password = st.text_input("Password:", type="password", key="password")

        submitted = st.form_submit_button("Login")

        if submitted:
            auth_result = authenticate(host, username, password)
            if isinstance(auth_result, tuple) and auth_result[0] is False:
                st.error(f"❌ {auth_result[1]}")
            else:
                st.session_state.authenticated_user = {"username": username}
                st.session_state.ceph_admin = auth_result[1]
                st.success(
                    f"✅ Logged in, {username}! Ceph Admin host: {st.session_state.ceph_admin.host}"
                )
                st.toast(f"Welcome {username}", icon="🎉")
                st.rerun()

# only if dashboard login is successfull
if st.session_state.authenticated_user:
    st.markdown(
        "<h1 style='text-align: center;'>🚀 Ceph Perf AI Agent</h1>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.2, 2], vertical_alignment="center", border=True)

    # 📊 Live Ceph Dashboard (col1 - Refreshes Automatically)
    with col1:
        st.markdown("##### 📊 Live Ceph Dashboard")

        ceph_status_container = st.container(key="live-ceph")
        live_services_container = st.container(key="live-services")

        def update_dashboard():
            ceph_status_json = ceph_ops.ceph_status(
                st.session_state.ceph_admin, json=True
            )
            ceph_status = parse_ceph_status(ceph_status_json)

            live_services_yaml = ceph_ops.get_live_services(st.session_state.ceph_admin)
            live_services = parse_live_services(live_services_yaml)

            # Update tables inside containers without affecting the rest of the UI
            with ceph_status_container:
                st.table(ceph_status)

            with live_services_container:
                st.table(live_services)

        update_dashboard()

        # Auto-refresh only col1 (every 5 seconds)
        st_autorefresh(interval=150000, key="Refresh-Live-Dashboard")

    # 💬 Ceph Chatbot (col2 - Static UI)
    with col2:
        st.markdown("##### 💬 Ceph Chatbot")
        with st.container(height=510, border=True, key="chatbot"):
            chat_display = st.container()  # Container to hold chat history
            chat_display.empty()

            # Display Chat History
            with chat_display:
                for _timestamp, _query, _response in st.session_state.chat_history:
                    st.write(_timestamp)
                    st.write(f"🧑‍💻 **You:** {_query}")

                    tabular_response = is_response_tabular(_response)
                    if tabular_response:
                        # Stream large text responses
                        st.write(f"🤖 **Bot:**  {tabular_response['general_summary']}")

                        # Show Performance Tuning Table
                        st.markdown("##### 📋 Performance Tuning Recommendations")
                        st.table(tabular_response["tabular_summary"][1])
                        st.write(tabular_response["tabular_summary"][0])
                    else:
                        st.markdown(f"🤖 **Bot:** {_response}")

                    st.divider()
            print(f"Number of history items: {len(st.session_state.chat_history)}")

            # User Input for Chatbot
            prompt = st.chat_input(
                "Hey, Choose your workload for performance enhancements and tunings..."
            )

            # Suggested Quick Actions
            for question in suggested_questions:
                if st.button(question, key=f"btn_{question}"):
                    prompt = question

            # Process Query
            if prompt:
                process_query(prompt)

                # Re-render chat history with updated messages
                # Display Chat History
                with chat_display:
                    # Re-render chat history with updated messages
                    chat_display.empty()

                    for _timestamp, _query, _response in st.session_state.chat_history:
                        st.write(_timestamp)
                        st.write(f"🧑‍💻 **You:** {_query}")

                        tabular_response = is_response_tabular(_response)
                        if tabular_response:
                            # Stream large text responses
                            st.write(
                                f"🤖 **Bot:**  {tabular_response['general_summary']}"
                            )

                            # Show Performance Tuning Table
                            st.markdown("##### 📋 Performance Tuning Recommendations")
                            st.table(tabular_response["tabular_summary"][1])
                            st.write(tabular_response["tabular_summary"][0])
                        else:
                            st.markdown(f"🤖 **Bot:** {_response}")

                        st.divider()

        # 🖥️ Execute Live Ceph Command
        with st.container(height=300, border=True):
            st.markdown("##### 🖥️ Execute Live Ceph Command")

            ceph_command = st.text_input("Enter a Ceph command >")
            if st.button("Run Command", key="run_ceph_cmd"):
                result = ceph_ops.execute_ceph_command(
                    st.session_state.ceph_admin, ceph_command
                )
                st.session_state.command_result = result
                st.code(result, language="bash")
