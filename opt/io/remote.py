import json
import uuid
import requests
import numpy as np


DOMAINS = {
    "dom.nri.can.va",
    "dom.nri.us.li",
    "dom.nri.can.ve",
    "dom.nri.can.ma",
    "dom.nri.can.tb",
    "dom.nri.us.la",
    "dom.nri.us.ld",
}

SERVER = "https://wayfinder-routing-request.playground-wayfinder-36.serverless.sandbox.verses.build"
#SERVER = "https://wayfinder.sandbox.verses.build"

def construct_hsml(domain_name):
    return {"resolved_by": {"name": domain_name}}


def construct_headers(hsml_json):
    return {
        "content-type": "application/json",
        "ce-type": "simflow.routing",
        "ce-specversion": "1.0",
        "ce-source": "/environment/production/1",
        "ce-id": "foo",
        "ce-channel": "activities.simflow.routing",
        "ce-correlationid": "foo",
        "ce-hsml": json.dumps(hsml_json),
    }


def format_connections(connections):
    return [{"_from": c["start"]["swid"], "_to": c["end"]["swid"], "distance": waypoint_dis(c)} for c in connections]


def format_waypoints(waypoints):
    return [format_waypoint(w) for w in waypoints]


def format_waypoint(w):
    w["name"] = "waypoint"
    return w


def waypoint_dis(c):
    start = np.array(list(c["start"]["coordinates"].values()))
    end = np.array(list(c["end"]["coordinates"].values()))
    return np.sqrt(np.sum((start - end) ** 2, axis=0))


def format_containers(spaces, dimensions, space_type="slot"):
    containers = dict()
    for space in spaces:
        space_type = space["state"]["space_type"]["type"]
        if space_type == space_type:
            name = space.get("name")
            coords = space.get("coordinates")
            location = []
            for dim in dimensions:
                location.append(coords.get(dim))
            containers[name] = tuple(location)
    return containers


def apply_constraints(spaces, constraints):
    res = []
    for space in spaces:
        for k, v in constraints.items():
            valid = True
            if space["coordinates"][k] < v[0] or space["coordinates"][k] > v[1]:
                valid = False
            if valid:
                res.append(space)
    return res


def apply_constraints_to_connections(connections, constraints):
    res = []
    for conn in connections:
        for k, v in constraints.items():
            valid = True
            if (
                conn["start"]["coordinates"][k] < v[0]
                or conn["start"]["coordinates"][k] > v[1]
                or conn["end"]["coordinates"][k] < v[0]
                or conn["end"]["coordinates"][k] > v[1]
            ):
                valid = False
            if valid:
                res.append(conn)
    return res


def get_warehouse_data(domain_name, constraints=None):
    # TODO
    # constraints = {"z": (2, 10)}
    print("Getting warehouse data via cloud events", domain_name)
    hsml_json = construct_hsml(domain_name)
    headers = construct_headers(hsml_json)
    payload = {
        "waypointArgs": {
            "returnWaypoints": True,
        }
    }

    r = requests.post(SERVER, data=json.dumps(payload), headers=headers)
    res = r.json()
    waypoints = res["data"]["waypoints"]["nodes"]
    connections = res["data"]["waypoints"]["edges"]

    if constraints:
        waypoints = apply_constraints(waypoints, constraints)
        connections = apply_constraints_to_connections(connections, constraints)

    connections = format_connections(connections)
    waypoints = format_waypoints(waypoints)
    connections = {uuid.uuid4(): v for v in connections}
    return waypoints, connections


def get_space_locations(domain_name, dimensions=["x", "y", "z"], constraints=None):
    # constraints = {"z": (4, 10)}
    hsml_json = construct_hsml(domain_name)
    headers = construct_headers(hsml_json)
    payload = {
        "queryArgs": {"cullSpaces": False},
    }

    r = requests.post(SERVER, data=json.dumps(payload), headers=headers)
    res = r.json()
    spaces = res["data"]["spaces"]["query"]
    if constraints:
        spaces = apply_constraints(spaces, constraints)

    containers = format_containers(spaces, dimensions)
    return containers


if __name__ == "__main__":
    domain_id = "dom.nri.us.ld"
    waypoints, connections = get_warehouse_data(domain_id, constraints={"z": (2, 10)})
    print(f"got {len(waypoints)} waypoints and {len(connections)} connections")
    containers = get_space_locations(domain_id, dimensions=["x", "y", "z"], constraints={"z": (4, 10)})
    print(f"got {len(containers)} containers")
