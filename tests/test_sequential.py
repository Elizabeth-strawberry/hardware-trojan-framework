import unittest
from trojan_refinement.sequential import analyze_sequential_modules as analyze

LIB = {
    "ff": {
        "directions": {"DIN": "input", "CLK": "input", "Q": "output", "EB": "input"},
        "sequential": True,
        "definition": "p_ff internal",
    },
    "xor": {
        "directions": {"A": "input", "B": "input", "Q": "output"},
        "sequential": False,
        "definition": "xor gate(Q,A,B);",
    },
    "and": {
        "directions": {"A": "input", "B": "input", "Q": "output"},
        "sequential": False,
        "definition": "and gate(Q,A,B);",
    },
}


def cell(t, **ports):
    return {"cell_type": t, "port_connections": ports}


class SequentialTests(unittest.TestCase):

    def circuit(self):
        return {
            "r": cell("ff", DIN="next", CLK="clk", Q="state", EB="en"),
            "update": cell("xor", A="state", B="en", Q="next"),
            "enable": cell("and", A="a", B="b", Q="en"),
            "decode": cell("and", A="state", B="flag", Q="hit"),
            "out": cell("ff", DIN="hit", CLK="clk", Q="latched"),
            "payload": cell("xor", A="latched", B="host", Q="result"),
            "other": cell("ff", DIN="input", CLK="clk", Q="unrelated"),
        }

    def test_modules_and_clock_isolation(self):
        g = analyze(self.circuit(), {"r", "enable"}, LIB)["groups"][0]
        self.assertEqual(g["state_registers"], ["r"])
        self.assertEqual(g["modules"]["state_core"]["instances"], ["r", "update"])
        self.assertEqual(g["modules"]["enable_logic"]["instances"], ["enable"])
        self.assertEqual(g["modules"]["state_decoder"]["instances"], ["decode"])
        self.assertNotIn("other", g["candidate_instances"])
        self.assertEqual(g["type_hypothesis"], "fsm")
        self.assertFalse(g["type_confirmed"])

    def test_renaming_does_not_change_type(self):
        c = self.circuit()
        renamed = {f"anonymous_{i}": v for i, v in enumerate(c.values())}
        a = analyze(c, {"r"}, LIB)["groups"][0]
        b = analyze(renamed, {"anonymous_0"}, LIB)["groups"][0]
        self.assertEqual(a["type_hypothesis"], b["type_hypothesis"])
        self.assertEqual(len(a["candidate_instances"]), len(b["candidate_instances"]))

    def test_instance_budget(self):
        g = analyze(self.circuit(), {"r"}, LIB, max_instances=3)["groups"][0]
        self.assertLessEqual(len(g["candidate_instances"]), 3)
        self.assertEqual(g["completeness"], "partial")

    def test_no_state_feedback(self):
        c = {"a": cell("and", A="x", B="y", Q="z")}
        self.assertEqual(analyze(c, {"a"}, LIB)["groups"], [])

    def test_feedback_fsm_hypothesis(self):
        c = {
            "r": cell("ff", DIN="a", CLK="clk", Q="q"),
            "s": cell("ff", DIN="b", CLK="clk", Q="p"),
            "a": cell("and", A="p", B="x", Q="a"),
            "b": cell("and", A="q", B="y", Q="b"),
        }
        g = analyze(c, {"r"}, LIB)["groups"][0]
        self.assertEqual(g["type_hypothesis"], "fsm")
        self.assertFalse(g["type_confirmed"])

    def test_longest_forward_chain(self):
        c = {
            "r": cell("ff", DIN="in", CLK="clk", Q="q"),
            "short": cell("ff", DIN="q", CLK="clk", Q="short_q"),
            "s": cell("ff", DIN="q", CLK="other_clk", Q="p"),
            "t": cell("ff", DIN="p", CLK="clk", Q="end"),
        }
        g = analyze(c, {"r"}, LIB, structure_type="counter")["groups"][0]
        self.assertEqual(g["dff_chain"], ["r", "s", "t"])

    def test_payload_xor_allows_consumers_but_stops_expansion(self):
        c = self.circuit()
        c["consumer"] = cell("and", A="result", B="x", Q="used")
        result = analyze(c, {"r"}, LIB)
        g = result["groups"][0]
        self.assertEqual(g["modules"]["state_decoder"]["instances"], ["decode"])
        self.assertNotIn("consumer", g["candidate_instances"])
        payload = g["payload_connections"][0]
        self.assertEqual(payload["instance_name"], "payload")
        self.assertEqual(
            payload["downstream_connections"],
            [{"signal": "result", "target": "consumer", "target_port": "A"}],
        )
        self.assertEqual(payload["other_inputs"], [{"port": "B", "signal": "host", "drivers": []}])

    def test_xor_must_directly_follow_output_register(self):
        c = self.circuit()
        c["buffer"] = cell("and", A="latched", B="x", Q="buffered")
        c["payload"]["port_connections"]["A"] = "buffered"
        self.assertTrue(
            all((not g["payload_connections"] for g in analyze(c, {"r"}, LIB)["groups"]))
        )

    def test_internal_paths_exclude_side_inputs(self):
        c = self.circuit()
        c["side"] = cell("and", A="x", B="y", Q="flag")
        g = analyze(c, {"r"}, LIB)["groups"][0]
        self.assertEqual(g["modules"]["state_decoder"]["instances"], ["decode"])
        self.assertNotIn("side", g["candidate_instances"])

    def test_fsm_condition_cone_from_data_side_input_without_enable_pin(self):
        c = self.circuit()
        del c["r"]["port_connections"]["EB"]
        g = analyze(c, {"r", "enable"}, LIB)["groups"][0]
        self.assertEqual(g["modules"]["enable_logic"]["instances"], ["enable"])
        self.assertEqual(g["boundaries"]["enable_port_signals"], [])
        self.assertEqual(g["boundaries"]["state_update_condition_signals"], ["en"])

    def test_cycle_and_register_limit(self):
        c = {"r": cell("ff", DIN="p", CLK="clk", Q="q"), "s": cell("ff", DIN="q", CLK="clk", Q="p")}
        g = analyze(c, {"r"}, LIB, max_registers=1, structure_type="counter")["groups"][0]
        self.assertEqual(g["dff_chain"], ["r"])
        self.assertIn("register_budget_reached", g["stop_reasons"])

    def test_counter_core_and_decoder(self):
        g = analyze(self.circuit(), {"r"}, LIB, structure_type="counter")["groups"][0]
        self.assertEqual(set(g["modules"]), {"state_core", "state_decoder", "enable_logic"})
        self.assertEqual(g["modules"]["state_core"]["role"], "counter")
        self.assertEqual(g["modules"]["state_core"]["instances"], ["r", "update"])
        self.assertEqual(g["modules"]["state_decoder"]["instances"], ["decode"])
        self.assertIn("enable", g["candidate_instances"])
        self.assertEqual(g["type_selection"], "caller_requested")
        self.assertFalse(g["type_confirmed"])

    def test_fsm_core_includes_update(self):
        g = analyze(self.circuit(), {"r"}, LIB, structure_type="fsm")["groups"][0]
        self.assertEqual(g["modules"]["state_core"]["role"], "fsm")
        self.assertEqual(g["modules"]["state_core"]["instances"], ["r", "update"])
        self.assertNotIn("state_update_logic", g["modules"])

    def test_invalid_structure_type(self):
        with self.assertRaises(ValueError):
            analyze(self.circuit(), {"r"}, LIB, structure_type="auto")

    def test_enable_stops_at_chain_q_and_other_branches_continue(self):
        c = self.circuit()
        c["enable"]["port_connections"].update(A="state", B="side_signal")
        c["side_gate"] = cell("and", A="primary_a", B="primary_b", Q="side_signal")
        g = analyze(c, {"r"}, LIB, structure_type="fsm")["groups"][0]
        self.assertEqual(g["modules"]["enable_logic"]["instances"], ["enable", "side_gate"])
        self.assertEqual(g["boundaries"]["chain_output_stop_signals"], ["state"])
        self.assertNotIn("update", g["modules"]["enable_logic"]["instances"])
        self.assertNotIn("r", g["modules"]["enable_logic"]["instances"])
        self.assertIn("primary_a", g["boundaries"]["external_inputs"])

    def test_fsm_scc_excludes_one_way_output_and_host(self):
        c = {
            "r": cell("ff", DIN="p", CLK="clk", Q="q"),
            "s": cell("ff", DIN="q", CLK="clk", Q="p"),
            "out": cell("ff", DIN="q", CLK="clk", Q="outq"),
            "host": cell("ff", DIN="outq", CLK="clk", Q="hostq"),
        }
        result = analyze(c, {"r", "s"}, LIB)
        self.assertEqual(len(result["groups"]), 1)
        g = result["groups"][0]
        self.assertEqual(g["state_component"], ["r", "s"])
        self.assertEqual(g["dff_chain"], [])
        self.assertEqual(g["modules"]["state_core"]["instances"], ["r", "s"])

    def test_fsm_oversized_scc_not_split(self):
        c = {"r": cell("ff", DIN="p", CLK="clk", Q="q"), "s": cell("ff", DIN="q", CLK="clk", Q="p")}
        r = analyze(c, {"r", "s"}, LIB, max_registers=1)
        self.assertEqual(r["groups"], [])
        self.assertEqual(len(r["rejected_state_components"]), 1)
        self.assertEqual(r["rejected_state_components"][0]["state_register_count"], 2)

    def test_enable_connection_does_not_merge_scc(self):
        c = self.circuit()
        c["r"]["port_connections"]["EB"] = "unrelated"
        g = analyze(c, {"r"}, LIB)["groups"][0]
        self.assertEqual(g["state_component"], ["r"])

    def test_payload_output_to_host_register_is_boundary(self):
        c = self.circuit()
        c["host_ff"] = cell("ff", DIN="result", CLK="clk", Q="host_state")
        g = analyze(c, {"r"}, LIB)["groups"][0]
        self.assertEqual(g["state_component"], ["r"])
        self.assertIn("out", g["auxiliary_output_instances"])
        self.assertNotIn("host_ff", g["candidate_instances"])
        self.assertEqual(
            g["payload_connections"][0]["downstream_connections"][0]["target"], "host_ff"
        )

    def test_fsm_enable_feedback_with_constant_data(self):
        c = {
            "r": cell("ff", DIN="1'b0", EB="en", CLK="clk", Q="state"),
            "e": cell("and", A="state", B="trigger", Q="en"),
        }
        g = analyze(c, {"r"}, LIB)["groups"][0]
        self.assertEqual(g["state_component"], ["r"])
        self.assertIn(
            {"source": "r", "target": "r", "edge_kind": "enable"}, g["state_dependency_edges"]
        )
        self.assertEqual(g["modules"]["enable_logic"]["instances"], ["e"])

    def test_mutual_enable_feedback_forms_scc(self):
        c = {
            "r": cell("ff", DIN="1'b0", EB="p", CLK="clk", Q="q"),
            "s": cell("ff", DIN="1'b0", EB="q", CLK="clk", Q="p"),
        }
        self.assertEqual(analyze(c, {"r"}, LIB)["groups"][0]["state_component"], ["r", "s"])

    def test_counter_does_not_use_enable_for_chain(self):
        c = {
            "r": cell("ff", DIN="in", EB="p", CLK="clk", Q="q"),
            "s": cell("ff", DIN="other", EB="q", CLK="clk", Q="p"),
        }
        self.assertEqual(
            analyze(c, {"r"}, LIB, structure_type="counter")["groups"][0]["dff_chain"], ["r"]
        )

    def test_output_dff_enable_cone_is_recovered(self):
        c = self.circuit()
        c["out"]["port_connections"]["EB"] = "output_en"
        c["output_enable"] = cell("and", A="state", B="trigger_en", Q="output_en")
        c["trigger"] = cell("and", A="external_a", B="external_b", Q="trigger_en")
        g = analyze(c, {"r"}, LIB)["groups"][0]
        self.assertEqual(g["state_component"], ["r"])
        self.assertEqual(
            g["modules"]["enable_logic"]["instances"], ["enable", "output_enable", "trigger"]
        )
        self.assertEqual(g["boundaries"]["enable_registers"], ["out", "r"])
        self.assertNotIn("r", g["modules"]["enable_logic"]["instances"])
        self.assertNotIn("update", g["modules"]["enable_logic"]["instances"])

    def test_counter_recovers_output_enable_module(self):
        c = self.circuit()
        c["out"]["port_connections"]["EB"] = "en"
        g = analyze(c, {"r"}, LIB, structure_type="counter")["groups"][0]
        self.assertEqual(g["modules"]["enable_logic"]["instances"], ["enable"])

    def test_counter_cross_bit_xor_is_not_output_boundary(self):
        c = {
            "r": cell("ff", DIN="rnext", CLK="clk", Q="q"),
            "rupdate": cell("and", A="q", B="trigger", Q="rnext"),
            "s": cell("ff", DIN="snext", CLK="clk", Q="p"),
            "carry_update": cell("xor", A="q", B="p", Q="snext"),
            "decode": cell("and", A="p", B="q", Q="hit"),
            "out": cell("ff", DIN="hit", CLK="clk", Q="outq"),
            "payload": cell("xor", A="outq", B="host", Q="result"),
        }
        r = analyze(c, {"r"}, LIB, structure_type="counter")
        g = r["groups"][0]
        self.assertEqual(g["dff_chain"], ["r", "s"])
        self.assertIn("carry_update", g["modules"]["state_core"]["instances"])
        self.assertEqual([p["instance_name"] for p in g["payload_connections"]], ["payload"])
        self.assertIn(
            {"instance_name": "carry_update", "feedback_destination_dffs": ["s"]},
            r["internal_state_xor_connections"],
        )

    def test_fsm_data_side_condition_does_not_absorb_state_feedback(self):
        c = {
            "r": cell("ff", DIN="next", CLK="clk", Q="q"),
            "update": cell("and", A="q", B="trigger", Q="next"),
            "trigger_gate": cell("and", A="a", B="b", Q="trigger"),
        }
        g = analyze(c, {"r"}, LIB)["groups"][0]
        self.assertEqual(g["state_component"], ["r"])
        self.assertEqual(g["modules"]["state_core"]["instances"], ["r", "update"])
        self.assertEqual(g["modules"]["enable_logic"]["instances"], ["trigger_gate"])
        self.assertNotIn("update", g["modules"]["enable_logic"]["instances"])

    def test_fsm_state_register_can_directly_drive_payload(self):
        c = {
            "r": cell("ff", DIN="next", CLK="clk", Q="q"),
            "update": cell("and", A="q", B="trigger", Q="next"),
            "payload": cell("xor", A="q", B="host", Q="result"),
            "host_ff": cell("ff", DIN="result", CLK="clk", Q="hostq"),
        }
        g = analyze(c, {"r"}, LIB)["groups"][0]
        self.assertEqual(g["state_component"], ["r"])
        self.assertEqual(g["modules"]["state_core"]["instances"], ["r", "update"])
        self.assertIn("payload", g["candidate_instances"])
        self.assertNotIn("host_ff", g["candidate_instances"])
        self.assertEqual(g["payload_connections"][0]["output_role"], "state_and_output_register")
        self.assertEqual(g["modules"]["state_decoder"]["instances"], [])

    def test_validation(self):
        with self.assertRaises(ValueError):
            analyze(self.circuit(), {"missing"}, LIB)
        with self.assertRaises(ValueError):
            analyze(self.circuit(), {"r"}, LIB, max_depth=0)
