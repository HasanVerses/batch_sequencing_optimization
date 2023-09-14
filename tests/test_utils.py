from opt.domain import NRI_DOMAINS, standardize_name, parse_domain_identifier
from opt.algorithms.common import convert_duplicate_node_multi, create_duplicate_node_multi
from opt.algorithms.reslotting import sd_from_bin_columns, bin_columns_from_sd
from opt.api.utils.reslotting import get_swap_batches
from opt.api.utils.snaking import get_pick_batches



def test_standardize_name():
    assert LI_SHORT == standardize_name(LI)
    assert LI_SHORT == standardize_name(LI_SWID)


def test_parse_domain_identifier():
    assert LI == parse_domain_identifier(LI_SWID, NRI_DOMAINS, return_reverse_domain_lookup=True)
    assert LI_SWID == parse_domain_identifier(LI, NRI_DOMAINS)


def test_duplicate_encoding():
    duplicates = create_duplicate_node_multi(sources)
    assert [n == len([x for x in duplicates if type(x) == str and f'{n}' in x]) for n in range(5)]
    assert sources == convert_duplicate_node_multi(duplicates)


def test_get_swap_batches():
    batches = get_swap_batches(bin_A, bin_B, swap_batch_size)
    assert swap_batch_size*2 == len(batches[0][0])
    assert swap_batch_remainder*2 == len(batches[-1][0])


def test_get_pick_batches():
    batches = get_pick_batches(bin_A, pick_batch_size)
    assert pick_batch_size == len(batches[0])
    assert pick_batch_remainder == len(batches[-1])


def test_swap_sd_maps():
    assert (bin_A, bin_B) == bin_columns_from_sd(*sd_from_bin_columns(bin_A, bin_B))
    assert sd_from_bin_columns(bin_A, bin_B) == sd_from_bin_columns(*bin_columns_from_sd(*sd_from_bin_columns(bin_A, bin_B)))


LI = "dom.nri.us.li"
LI_SHORT = "LI"
LI_SWID = "e20deb3b-2144-4061-b9d7-65d87994c64b"

sources = [557, 557, 690, 1000, 1000, 1000, 1000, 2, 4, 3, 3]

numbers = range(65, 91)
bin_A, bin_B = [chr(x) for x in numbers], list(numbers)
first_batch_range = range(65,75)
last_batch_range = range(85, 91)

swap_batch_size = 5
pick_batch_size = 10
pick_batch_remainder = len(numbers) % pick_batch_size
swap_batch_remainder = len(numbers) % swap_batch_size