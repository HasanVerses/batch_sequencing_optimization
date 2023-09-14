from opt.io.local import get_graphs
from opt.domain import ALL_DCS, NEW_DCS

# Store all usable DC graphs locally


if __name__ == "__main__":
    legacy = True
    dcs_to_use = ALL_DCS if legacy else NEW_DCS
    get_graphs(dcs_to_use=dcs_to_use)
