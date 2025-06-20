from copy import deepcopy

from .config import ceph_configs


class Performance:
    def load_config(self):
        """load ceph configs from backend.config"""
        return deepcopy(ceph_configs)

    def list_params(self, params):
        """Return list of ceph configs based on the required parameters."""
        ceph_configs = self.load_config()
        return [ceph_configs[param] for param in params if param in ceph_configs]

    def compare(self, comparator, op1, op2):
        comps = {
            "lt": lambda x, y: x < y,
            "gt": lambda x, y: x > y,
            "eq": lambda x, y: x == y,
            "in": lambda x, y: x in y,
        }
        return comps[comparator](op1, op2)

    def evaluate(self, client, config):
        """Evaluate ceph config setting and prepare summary.

        Example:

        "osd_memory_target": {
            "param": "osd_memory_target",
            "get_command": "ceph config get osd osd_memory_target",
            "set_command": "ceph config set osd osd_memory_target 8589934592",
            "expected": 8589934592,
            "effect": "Reduces OSD read/write latency.",
            "positive": "'osd_memory_target' is already more than 8GB.",
            "suggest": "Increase 'osd_memory_target' to more than 8GB",
            "comparator": "lt"
            },
        """
        get_cmd = config["get_command"]
        status, output = client.execute_command(command=config["get_command"])
        if not status:
            raise Exception(f"failed to execute this command : `{get_cmd}`")

        positive = self.compare(
            config["comparator"], output.strip(), config["expected"]
        )
        _tuning = {
            "Setting": config["param"],
            "Description": f"🔍 {config['effect']}",
        }

        if positive:
            _tuning["Status"] = "✅ Configured optimally"
            _tuning["Recommendation"] = "✔️ No changes needed"
            _tuning["Run Command"] = ""
        else:
            _tuning["Status"] = "⚠️ Not set"
            _tuning["Recommendation"] = f"🔧 {config['suggest']}"
            _tuning["Run Command"] = config["set_command"]

        return _tuning

    def summarize(self, client, params):
        """Summarize the suggestions for Peformance tuning."""
        _params = self.list_params(params)
        return [self.evaluate(client, i) for i in _params]
