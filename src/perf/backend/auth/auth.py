import sys

from .cephadmin import CephAdminClient

sys.path.append("..")


def authenticate(host, user, password):
    """Authenticate ceph admin or dashboard password."""
    client = None
    try:
        client = CephAdminClient(host, user, password)
        client.connect()
        print("authetication Successfull...")
        return True, client
    except Exception as e:
        if client:
            client.close()
        return False, f"Ceph admin node access failed: {e}"
