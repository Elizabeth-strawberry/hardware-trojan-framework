import unittest
from trojan_refinement.module_recovery import recover_missing_modules as recover

LIB = {
    "and": {"directions": {"A": "input", "B": "input", "Y": "output"}, "sequential": False},
    "xor": {"directions": {"A": "input", "B": "input", "Y": "output"}, "sequential": False},
    "inv": {"directions": {"A": "input", "Y": "output"}, "sequential": False},
    "dff": {
        "directions": {"D": "input", "EB": "input", "CLK": "input", "Q": "output"},
        "sequential": True,
    },
}


def cell(kind, **ports):
    return {"cell_type": kind, "port_connections": ports}


def seq_group(names, core):
    return {
        "kind": "fsm",
        "candidate_instances": names,
        "module_evidence": {
            "state_registers": core,
            "modules": {"state_core": {"instances": core}},
            "payload_connections": [],
        },
    }


class ModuleRecoveryTests(unittest.TestCase):

    def test_private_payload_input_recovered_and_shared_host_input_stopped(self):
        cells = {
            "hidden": cell("and", A="a", B="b", Y="trigger"),
            "host": cell("and", A="h1", B="h2", Y="host_signal"),
            "other": cell("inv", A="host_signal", Y="other_out"),
            "payload": cell("xor", A="trigger", B="host_signal", Y="out"),
        }
        r = recover(
            cells,
            {"payload"},
            [],
            [{"instance_name": "payload"}],
            LIB,
            "Comb.",
            depth=2,
            max_added=100,
        )
        self.assertEqual(r["new_scope_instances"], ["hidden"])
        self.assertTrue(any((b["reason"] == "shared_producer" for b in r["boundaries"])))

    def test_depth_and_new_instance_budget_are_enforced(self):
        cells = {
            "a": cell("inv", A="i", Y="a"),
            "b": cell("inv", A="a", Y="b"),
            "p": cell("xor", A="b", B="host", Y="out"),
        }
        r = recover(
            cells, {"p"}, [], [{"instance_name": "p"}], LIB, "Comb.", depth=1, max_added=100
        )
        self.assertEqual(r["new_scope_instances"], ["b"])
        r = recover(cells, {"p"}, [], [{"instance_name": "p"}], LIB, "Comb.", depth=2, max_added=1)
        self.assertEqual(r["new_scope_instances"], ["b"])
        self.assertTrue(r["budget_truncated"])

    def test_recover_enable_gate_but_not_upstream_register_or_clock_tree(self):
        cells = {
            "state": cell("dff", D="q", EB="enable", CLK="clk", Q="q"),
            "upstream": cell("dff", D="input", EB="one", CLK="clk", Q="source"),
            "enable": cell("and", A="q", B="source", Y="enable"),
            "clock_driver": cell("inv", A="clock_input", Y="clk"),
        }
        r = recover(cells, {"state"}, [seq_group(["state"], ["state"])], [], LIB, "FSM", depth=2)
        self.assertEqual(r["new_scope_instances"], ["enable"])
        self.assertTrue(any((b["reason"] == "external_register" for b in r["boundaries"])))

    def test_decoder_requires_output_dff_xor_and_does_not_follow_payload_into_host(self):
        cells = {
            "s": cell("dff", D="q", EB="one", CLK="clk", Q="q"),
            "decode": cell("inv", A="q", Y="decoded"),
            "out_reg": cell("dff", D="decoded", EB="one", CLK="clk", Q="out_q"),
            "xor": cell("xor", A="out_q", B="host_signal", Y="payload_out"),
            "host_gate": cell("and", A="payload_out", B="h", Y="host_out"),
        }
        r = recover(cells, {"s"}, [seq_group(["s"], ["s"])], [], LIB, "FSM", depth=2)
        self.assertEqual(set(r["new_scope_instances"]), {"decode", "out_reg", "xor"})

    def test_missing_fsm_register_requires_bounded_feedback(self):
        cells = {
            "s": cell("dff", D="missing_q", EB="one", CLK="clk", Q="q"),
            "missing": cell("dff", D="q", EB="one", CLK="clk", Q="missing_q"),
        }
        r = recover(cells, {"s"}, [seq_group(["s"], ["s"])], [], LIB, "FSM", depth=2)
        self.assertEqual(r["new_scope_instances"], ["missing"])
        cells["missing"]["port_connections"]["D"] = "unrelated_input"
        r = recover(cells, {"s"}, [seq_group(["s"], ["s"])], [], LIB, "FSM", depth=2)
        self.assertEqual(r["new_scope_instances"], [])

    def test_module_stage_can_complete_previously_missing_combinational_roles(self):
        from trojan_refinement.module_checker import check_modules as module_stage

        cells = {
            "left": cell("and", A="a", B="b", Y="l"),
            "right": cell("inv", A="c", Y="r"),
            "decision": cell("and", A="l", B="r", Y="trigger"),
            "payload": cell("xor", A="trigger", B="host", Y="out"),
        }

        def initial():
            return {
                "groups": [
                    {
                        "group_id": "p1",
                        "accepted": True,
                        "seed_instances": ["payload"],
                        "expanded_instances": ["payload"],
                        "payload_clues": [{"instance_name": "payload", "internal_inputs": {}}],
                    }
                ]
            }

        self.assertEqual(
            module_stage(cells, {"payload"}, initial(), "Comb.", LIB, recovery_depth=0), []
        )
        groups = module_stage(cells, {"payload"}, initial(), "Comb.", LIB, recovery_depth=2)
        self.assertEqual({n for g in groups for n in g["candidate_instances"]}, set(cells))
