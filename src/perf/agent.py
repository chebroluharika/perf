import json

import yaml
from langchain.tools import Tool

from .backend.ceph_operations import Ceph
from .backend.perfomance import Performance


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
            f"- **Total Tunable Profiles:** {total_settings}\n"
            f"- **Optimally Configured:** {optimally_configured}\n"
            f"- **Profiles Need Attention:** {needs_attention}\n"
        )

        if recommended_commands:
            summary += "\n- **Recommended Configuration Commands:**\n"
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


def get_ceph_status(ceph_admin):
    """fetches ceph cluster status of running cluster."""
    # if not st.session_state.authenticated_user:
    #     return "⚠️ Please log in first."
    # status = ceph_ops.ceph_status(st.session_state.ceph_admin)
    status = ceph_ops.ceph_status(ceph_admin)
    return status if status else "Failed to retrieve Ceph status."


def recommend_perf_tunables_low_latency_dbs(status):
    """
    Performance recommendations for Low latency databases
    like MySQL, PostgreSQL, MongoDB and other databases.
    """
    key = "Low Latencys databases workload"
    # if not st.session_state.authenticated_user:
    #     return "⚠️ Please log in first."
    # status = perf_ops.summarize(
    #     client=st.session_state.ceph_admin, params=USE_CASES["LOW_LATENCY_DBS_WORKLOAD"]
    # )
    tabular_summary = tabulate_summary(key, status)
    # general_summary = generate_summary(key, status)

    response = {
        # "general_summary": general_summary,
        "tabular_summary": tabular_summary,
    }
    return (
        json.dumps(response)
        if response
        else f"Failed to retrieve perf recommendations tunbales for {key}."
    )


def recommend_perf_tunables_high_throughput(status):
    """
    Performance recommendations for High throughput workloads
    like Media Streaming, Backup Storage.
    """
    key = "High throughput workload"
    # if not st.session_state.authenticated_user:
    #     return "⚠️ Please log in first."
    # status = perf_ops.summarize(
    #     client=st.session_state.ceph_admin,
    #     params=USE_CASES["HIGH_THROUGHPUT_WORKLOADS"],
    # )

    tabular_summary = tabulate_summary(key, status)
    # general_summary = generate_summary(key, status)

    response = {
        # "general_summary": general_summary,
        "tabular_summary": tabular_summary,
    }
    return (
        json.dumps(response)
        if response
        else f"Failed to retrieve perf recommendations tunbales for {key}."
    )


def recommend_perf_tunables_vm_storage(status):
    """
    Performance recommendations for Virtualization environment
    workloads like OpenStack, KVM, Proxmox.
    """
    key = "Vitualization environment workload"
    # if not st.session_state.authenticated_user:
    #     return "⚠️ Please log in first."
    # status = perf_ops.summarize(
    #     client=st.session_state.ceph_admin, params=USE_CASES["VIRTUAL_MACHINE_STORAGE"]
    # )

    tabular_summary = tabulate_summary(key, status)
    # general_summary = generate_summary(key, status)

    response = {
        # "general_summary": general_summary,
        "tabular_summary": tabular_summary,
    }
    return (
        json.dumps(response)
        if response
        else f"Failed to retrieve perf recommendations tunbales for {key}."
    )


def recommend_perf_tunables_big_data(status):
    """
    Performance recommendations for Big Data workloads
    like Hadoop, Spark, Elasticsearch.
    """
    key = "Big Data environment workload"
    # if not st.session_state.authenticated_user:
    #     return "⚠️ Please log in first."
    # status = perf_ops.summarize(
    #     client=st.session_state.ceph_admin, params=USE_CASES["BIG_DATA_ANALYTICS"]
    # )

    tabular_summary = tabulate_summary(key, status)
    # general_summary = generate_summary(key, status)

    response = {
        # "general_summary": general_summary,
        "tabular_summary": tabular_summary,
    }
    return (
        json.dumps(response)
        if response
        else f"Failed to retrieve perf recommendations tunbales for {key}."
    )


def recommend_perf_tunables_object_workloads(status):
    """Performance recommendations for Object workloads."""
    key = "Object workloads"
    # if not st.session_state.authenticated_user:
    #     return "⚠️ Please log in first."
    # status = perf_ops.summarize(
    #     client=st.session_state.ceph_admin, params=USE_CASES["OBJECT_WORKLOADS"]
    # )

    tabular_summary = tabulate_summary(key, status)
    # general_summary = generate_summary(key, status)

    response = {
        # "general_summary": general_summary,
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
                "Running": f"{service.get('status', {}).get('running', 0)!s}/{service.get('status', {}).get('size', 0)!s}",
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

# Use all the tools

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
    Tool(
        name="Get performance recommendations for object workloads",
        func=recommend_perf_tunables_object_workloads,
        description="Retrieve perf recommendations tunbales for Object Workloads.",
        return_direct=True,
    ),
]
