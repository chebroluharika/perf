"""Ceph Module."""


class Ceph:

    def execute_ceph_command(self, client, command):
        rc, output = client.execute_command(command=command)
        if rc:
            return output
        return f"Command Execution Failed: {output}"

    def ceph_status(self, client, json=False):
        command = "ceph -s"
        if json:
            command += " -f json"
        return self.execute_ceph_command(client, command=command)

    def get_available_hosts(self, client):
        return self.execute_ceph_command(client, command="ceph orch host ls -f yaml")

    def get_live_services(self, client):
        return self.execute_ceph_command(client, command="ceph orch ls -f yaml")
