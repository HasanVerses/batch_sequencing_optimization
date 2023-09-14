import json
import requests

from typing import Optional, Tuple
from opt.domain import NRI_DOMAINS, parse_domain_identifier



SERVER = "https://sandbox.cosm.nri.verses.io/graphql"

query_template = """  
{{
    domain(swid: "{uid}") {{
        space(swid: "", filters: {{
            component: "space_type",
            field: "type",
            value: "waypoint"
        }}){{
            swid
            name
            coordinates
    }}
  }}
}}
"""

space_query_template = """  
{{
    domain(swid: "{uid}") {{
        space(swid: ""){{
            swid
            name
            coordinates
    }}
  }}
}}
"""


def _space_query(domain_id: str) -> dict:
    domain_id = parse_domain_identifier(domain_id)
    body = space_query_template.format(uid=domain_id)
    ret = requests.post(SERVER, json={"query": body})

    return _validate_waypoint_query(ret)


def _waypoint_query(domain_id: str) -> dict:
    domain_id = parse_domain_identifier(domain_id)
    body = query_template.format(uid=domain_id)
    ret = requests.post(SERVER, json={"query": body})

    return _validate_waypoint_query(ret)


def _parse_errors(resp: str) -> None:
    response_json = json.loads(resp.text)
    print(f"Response status code {resp.status_code}")
    errors = response_json.get("errors")
    if errors:
        message = errors[0].get("message")
    print(message)

    return None


def _validate_waypoint_query(resp: dict) -> Optional[dict]:
    if resp.status_code == 200:
        data = json.loads(resp.text).get("data")
        assert data, "`data` field missing from response!"
        domain_data = data.get("domain")
        assert domain_data, "`domain` field empty or missing from response!"
        space_data = domain_data[0].get("space")
        assert space_data, "`space` field missing from response!"

        return space_data
    else:
        return _parse_errors(resp.text)
    

def _validate_connection_query(resp: dict) -> Optional[dict]:
    if resp.status_code == 200:
        return json.loads(resp.text)
    else:
        return _parse_errors(resp.text)


def _get_connections(url: str) -> Optional[dict]:
    ret = requests.get(url)
    return _validate_connection_query(ret)


def _connection_query(domain_id: str, domain_dict: dict=NRI_DOMAINS) -> Optional[dict]:
    domain_id = parse_domain_identifier(domain_id, return_reverse_domain_lookup=True)
    connections_url = domain_dict[domain_id]["connections_url"]

    return _get_connections(connections_url)


def get_warehouse_data(domain_id: str, domain_dict: dict=NRI_DOMAINS, constraints: None = None) -> Optional[Tuple[dict, dict]]:
    """
    Get waypoints and connections for a specified warehouse from the verses NRI COSM database

    Parameters: 
        `domain_id`: domain's SWID or reverse domain lookup identifier (e.g. `dom.nri.us.li`)

    Returns: 
        (waypoints, connections)

        where
        `waypoints` is a list of dicts containing data for individual waypoints, and
        `connections` is a dict of dicts containing edge (from_vertex, to_vertex and distance) data        
    """
    waypoints = _waypoint_query(domain_id)
    if not waypoints:
        print(f"Error obtaining waypoints for domain {domain_id}.")
        return None
    connections = _connection_query(domain_id, domain_dict)
    if not connections:
        print(f"Error obtaining connections for domain {domain_id}.")
        return None
    
    return waypoints, connections

def get_space_locations(domain_id, dimensions=["x", "y"]):
    space_info = _space_query(domain_id)
    containers = dict()
    for space in space_info:
        name = space.get("name")
        if not name or name[:3] != 'VA-':
            continue
        coords = space.get("coordinates")
        if not coords:
            continue
        location = []
        for dim in dimensions:
            location.append(coords.get(dim))
        containers[name] = tuple(location)

    return containers
