WAREHOUSES = { 
    # "LA": 'dom.nri.us.la',
    # "LD": 'dom.nri.us.ld',
    "LI": 'dom.nri.us.li',
    "VA": 'dom.nri.can.va',
    "VE": 'dom.nri.can.ve',
    "VB": 'dom.nri.can.vb',
    "KF": 'dom.nri.can.kf',
    "KB": 'dom.nri.can.kb',
    "MA": 'dom.nri.can.ma',
    "TB": 'dom.nri.can.tb'
}

NEW_DCS = {
    "LA": 'dom.nri.us.la',
    "LD": 'dom.nri.us.ld',
    "LI": 'dom.nri.us.li',
    "VA": 'dom.nri.can.va',
    "VE": 'dom.nri.can.ve',
    # "VB": 'dom.nri.can.vb',
    # "KF": 'dom.nri.can.kf',
    # "KB": 'dom.nri.can.kb',
    "MA": 'dom.nri.can.ma',
    "TB": 'dom.nri.can.tb'
}

ALL_DCS = list(WAREHOUSES.keys())

NRI_DOMAINS = {
    'dom.nri.us.li': {
        "swid": 'e20deb3b-2144-4061-b9d7-65d87994c64b', 
        "connections_url": 'https://cdn.verses.io/staticgeometry/nri/us/li/connections.json'
    },
    'dom.nri.can.va': {
        "swid": '259ac0ec-3cf5-49ed-9622-d1a0c10519a2',
        "connections_url": 'https://cdn.verses.io/staticgeometry/nri/can/va/connections.json'
    },
    'dom.nri.can.ve': {
        "swid": '458c1f26-3da8-4dbe-96c9-67c2f612d0d7',
        "connections_url": 'https://cdn.verses.io/staticgeometry/nri/can/ve/connections.json'
    },
    'dom.nri.can.vb': {
        "swid": 'cfca370a-03a4-4887-a8a2-c80ca95276c0',
        "connections_url": 'https://cdn.verses.io/staticgeometry/nri/can/vb/connections.json'
    },

    # Missing data
    # 'dom.nri.can.kc': {
    #     "swid": 'f037c4d5-60a3-4b6e-a65d-6c591e464d78',
    #     "connections_url": ''
    # },
    'dom.nri.can.kf': {
        "swid": '5a86a579-89d2-4de3-ae1e-1124e78d83cd',
        "connections_url": 'https://cdn.verses.io/staticgeometry/nri/can/kf/connections.json'
    },
    'dom.nri.can.kb': {
        "swid": '5327defb-7cd2-4fe6-91bd-6b443c392482',
        "connections_url": 'https://cdn.verses.io/staticgeometry/nri/can/kb/connections.json'
    },

    # Missing data
    # 'dom.nri.us.le': {
    #     "swid": 'a26458bc-b99d-44fd-8445-3bdd624472c0',
    #     "connections_url": ''
    # },
    # 'dom.nri.us.lh': {
    #     "swid": '7186660d-f91f-4a9c-ad69-11568f4ab372',
    #     "connections_url": 'https://cdn.verses.io/staticgeometry/nri/us/lh/connections.json'
    # },
    'dom.nri.can.ma': {
        "swid": 'ae79e5f4-5d11-4e92-a116-9448bcec252b',
        "connections_url": 'https://cdn.verses.io/staticgeometry/nri/can/ma/connections.json'
    },
    'dom.nri.can.tb': {
        "swid": '2b0af0a6-5207-49f7-a772-fa42a414e89c',
        "connections_url": 'https://cdn.verses.io/staticgeometry/nri/can/tb/connections.json'
    }
}


def parse_domain_identifier(domain_id: str, domain_dict: dict=NRI_DOMAINS, return_reverse_domain_lookup=False) -> str:
    assert domain_id[:3] == 'dom' or len(domain_id) == 36, \
    "Please provide either a reverse domain lookup string (e.g. 'dom.nri.us.li') or a 32-digit SWID"
    if domain_id[:3] == 'dom':
        if return_reverse_domain_lookup:
            return domain_id
        assert domain_id in domain_dict, "Domain identifier not recognized."
        domain_id = domain_dict[domain_id]["swid"]
    else:
        if return_reverse_domain_lookup:
            swids = {domain_dict[k]["swid"]: k for k in domain_dict.keys()}
            assert domain_id in swids, "Domain SWID not recognized."
            return swids[domain_id]

    return domain_id


def standardize_name(name: str):
    if len(name) > 2:   # Hack: check for abbreviated (e.g. "LA") format
        name = parse_domain_identifier(name, return_reverse_domain_lookup=True)
        name = name.split(".")[-1].upper()
    
    return name


def parse_bin_label(bin_label, reject_nonstandard=False):
    bin_label_components = bin_label.split('-')
    if bin_label_components[-1].isalpha() and reject_nonstandard:
        assert len(bin_label_components) == 5, "Cannot parse location name!"
    
    return bin_label_components
