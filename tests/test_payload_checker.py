import unittest
from trojan_refinement.payload_checker import check_payloads as check, find_payload_clues as payloads
from trojan_refinement.graph import build_graph_index as graph_index


def cell(kind, inputs, output):
    return {
        "cell_type": kind,
        "port_connections": {**{f"DIN{i}": s for i, s in enumerate(inputs)}, "Q": output},
    }


class PayloadCheckerTests(unittest.TestCase):

    def test_multiple_payloads_and_tolerance(self):
        cells = {
            "a": cell("xor2s1", ["i", "j"], "a"),
            "b": cell("xnr2s1", ["k", "l"], "b"),
            "c": cell("and2s1", ["a", "b"], "c"),
        }
        f, r, _, _ = graph_index(cells)
        self.assertEqual(payloads(set(cells), cells, f, r, 0), [])
        self.assertEqual(
            {x["instance_name"] for x in payloads(set(cells), cells, f, r, 1)}, {"a", "b"}
        )

    def test_expansion_does_not_reverse_direction(self):
        cells = {
            "s": cell("and2s1", ["p", "i"], "s"),
            "p": cell("and2s1", ["j", "k"], "p"),
            "branch": cell("xor2s1", ["p", "z"], "branch"),
            "d": cell("xor2s1", ["s", "m"], "d"),
        }
        result = check(cells, {"s"}, 2, 2, 0)
        self.assertEqual(set(result["candidate_instances"]), {"s", "p", "d"})

    def test_register_crossing_is_explicit(self):
        cells = {
            "s": cell("and2s1", ["i", "j"], "s"),
            "r": cell("dffs2", ["s"], "r"),
            "x": cell("xor2s1", ["r", "k"], "x"),
        }
        self.assertEqual(
            check(cells, {"s"}, 0, 2, 0, stop_at_registers=True)["candidate_instances"], []
        )
        self.assertEqual(
            set(check(cells, {"s"}, 0, 2, 0, stop_at_registers=False)["candidate_instances"]),
            set(cells),
        )

    def test_downstream_stop_does_not_extend_upstream_budget(self):
        cells = {
            "p": cell("and2s1", ["i", "j"], "p"),
            "s": cell("and2s1", ["p", "k"], "s"),
            "x": cell("xor2s1", ["s", "m"], "x"),
            "host": cell("and2s1", ["x", "n"], "h"),
        }
        result = check(cells, {"s"}, 2, 5, 0)
        self.assertNotIn("host", result["candidate_instances"])
        self.assertEqual(len(result["groups"][0]["expansion_steps"]), 2)
