import paramiko
from paramiko.ssh_exception import SSHException


class CephAdminClient:
    def __init__(self, host, username, password):
        self.host = host
        self.username = username
        self.password = password
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._auth = None

    def connect(self):
        try:
            self._auth = self.client.connect(
                hostname=self.host,
                username=self.username,
                password=self.password,
                timeout=60,
            )
        except SSHException as e:
            print("SSH connection failed:", str(e))
            raise paramiko.AuthenticationException(e)

    def execute_command(self, command):
        try:
            _, stdout, stderr = self.client.exec_command(command, timeout=120)
            output = stdout.read().decode()
            error = stderr.read().decode()

            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                print(
                    f"❌ Command failed with exit status {exit_status}. Error:", error
                )
                return False, error
            else:
                print("✅ Command executed successfully!, Output:", output)
                return True, output
        except Exception as e:
            print("Command execution failed:", str(e))
            return False, "Command execution failed:", str(e)

    def close(self):
        self.client.close()
