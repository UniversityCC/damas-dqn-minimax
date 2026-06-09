import unittest
from src.damas.engine import (
    initial_state, legal_moves, step, is_terminal, result,
    _JUMP_OVER, NEIGHBORS, empty_state_for_test,
)


class TestMandatoryCapture(unittest.TestCase):
    def test_only_captures_when_available(self):
        """Si hay captura disponible, legal_moves devuelve solo capturas."""
        s = empty_state_for_test(turn=1)
        found = False
        for a in range(32):
            for d in ["ul", "ur", "dl", "dr"]:
                mid = NEIGHBORS[a][d]
                if mid is None:
                    continue
                land = NEIGHBORS[mid].get(d)
                if land is None or land == a:
                    continue
                s["board"][a]    =  1
                s["board"][mid]  = -1
                found = True
                break
            if found:
                break
        self.assertTrue(found, "No se encontró posición de captura")
        moves = legal_moves(s)
        self.assertGreater(len(moves), 0)
        for m in moves:
            self.assertIn(
                (m[0], m[1]), _JUMP_OVER,
                f"Movimiento {m} no es captura, pero hay capturas disponibles"
            )

    def test_no_simple_move_alongside_capture(self):
        """No debe mezclarse un movimiento simple con capturas en la lista."""
        s = empty_state_for_test(turn=1)
        nb = NEIGHBORS[21]
        mid  = nb["ul"]
        if mid is None:
            self.skipTest("geometría no disponible")
        land = NEIGHBORS[mid].get("ul")
        if land is None:
            self.skipTest("geometría no disponible")
        s["board"][21]  =  1
        s["board"][mid] = -1
        moves = legal_moves(s)
        simple = [(m[0], m[1]) for m in moves if (m[0], m[1]) not in _JUMP_OVER]
        self.assertEqual(simple, [], f"Movimientos simples devueltos con captura: {simple}")


class TestMultiCapture(unittest.TestCase):
    def _chain_state(self):
        """Devuelve (estado, a, mid1, b, mid2, c) para doble captura."""
        s = empty_state_for_test(turn=1)
        for a in range(32):
            for d1 in ["ul", "ur", "dl", "dr"]:
                mid1 = NEIGHBORS[a][d1]
                if mid1 is None:
                    continue
                b = NEIGHBORS[mid1].get(d1)
                if b is None or b == a:
                    continue
                for d2 in ["ul", "ur", "dl", "dr"]:
                    mid2 = NEIGHBORS[b][d2]
                    if mid2 is None or mid2 == mid1:
                        continue
                    c = NEIGHBORS[mid2].get(d2)
                    if c is None or c in (a, b, mid1):
                        continue
                    s["board"][a]    =  1
                    s["board"][mid1] = -1
                    s["board"][mid2] = -1
                    return s, a, mid1, b, mid2, c
        return None

    def test_double_capture_action_present(self):
        res = self._chain_state()
        if res is None:
            self.skipTest("No se encontró cadena doble")
        s, a, mid1, b, mid2, c = res
        moves = legal_moves(s)
        self.assertIn((a, b, c), moves, f"Cadena ({a},{b},{c}) no en {moves}")

    def test_double_capture_removes_both(self):
        res = self._chain_state()
        if res is None:
            self.skipTest("No se encontró cadena doble")
        s, a, mid1, b, mid2, c = res
        s2 = step(s, (a, b, c))
        self.assertEqual(s2["board"][mid1], 0, "mid1 no eliminado")
        self.assertEqual(s2["board"][mid2], 0, "mid2 no eliminado")
        self.assertEqual(s2["board"][a],    0)
        self.assertNotEqual(s2["board"][c], 0)

    def test_capture_resets_no_capture_count(self):
        res = self._chain_state()
        if res is None:
            self.skipTest("No se encontró cadena doble")
        s, a, mid1, b, mid2, c = res
        s["no_capture_count"] = 20
        s2 = step(s, (a, b, c))
        self.assertEqual(s2["no_capture_count"], 0)


class TestPromotionStopsChain(unittest.TestCase):
    def test_promotion_stops_chain(self):
        s = empty_state_for_test(turn=1)
        crown_squares = {28, 29, 30, 31}
        for a in range(32):
            for d in ["dl", "dr"]:
                mid1 = NEIGHBORS[a].get(d)
                if mid1 is None:
                    continue
                crown = NEIGHBORS[mid1].get(d)
                if crown not in crown_squares:
                    continue
                for d2 in ["ul", "ur", "dl", "dr"]:
                    mid2 = NEIGHBORS[crown].get(d2)
                    if mid2 is None:
                        continue
                    beyond = NEIGHBORS[mid2].get(d2)
                    if beyond is None or beyond == a:
                        continue
                    s["board"][a]     =  1
                    s["board"][mid1]  = -1
                    s["board"][mid2]  = -1
                    moves = legal_moves(s)
                    self.assertNotIn(
                        (a, crown, beyond), moves,
                        "La cadena continúa más allá de la coronación"
                    )
                    self.assertIn(
                        (a, crown), moves,
                        "La captura que corona no aparece"
                    )
                    return
        self.skipTest("No se encontró posición de coronación en cadena")


class TestDrawByMoveCount(unittest.TestCase):
    def test_not_terminal_at_79(self):
        s = empty_state_for_test(turn=1)
        s["board"][10] =  1
        s["board"][20] = -1
        s["no_capture_count"] = 79
        self.assertFalse(is_terminal(s))

    def test_terminal_at_80(self):
        s = initial_state()
        s["no_capture_count"] = 80
        self.assertTrue(is_terminal(s))

    def test_result_draw_at_80(self):
        s = initial_state()
        s["no_capture_count"] = 80
        self.assertEqual(result(s), 0)

    def test_step_increments_counter_simple(self):
        s = empty_state_for_test(turn=1)
        s2_base = initial_state()
        moves = legal_moves(s2_base)
        m = moves[0]
        s2 = step(s2_base, m)
        self.assertEqual(s2["no_capture_count"], 1)

    def test_step_resets_counter_on_capture(self):
        s = empty_state_for_test(turn=1)
        s["no_capture_count"] = 35
        for a in range(32):
            for d in ["ul","ur","dl","dr"]:
                mid = NEIGHBORS[a][d]
                if mid is None: continue
                land = NEIGHBORS[mid].get(d)
                if land is None: continue
                s["board"][a]   =  1
                s["board"][mid] = -1
                s2 = step(s, (a, land))
                self.assertEqual(s2["no_capture_count"], 0)
                return
        self.skipTest("Sin salto disponible")


class TestThreefoldRepetition(unittest.TestCase):
    def test_result_none_before_third(self):
        s = empty_state_for_test(turn=1)
        s["board"][15] = 2
        key = (tuple(s["board"]), 1)
        s["position_history"][key] = 2
        self.assertIsNone(result(s))

    def test_result_draw_on_third(self):
        s = empty_state_for_test(turn=1)
        s["board"][15] = 2
        key = (tuple(s["board"]), 1)
        s["position_history"][key] = 3
        self.assertTrue(is_terminal(s))
        self.assertEqual(result(s), 0)

    def test_threefold_via_step(self):
        s = empty_state_for_test(turn=1)
        nb10 = NEIGHBORS[10]
        nb22 = NEIGHBORS[22]
        dest_r = nb10.get("dr")
        dest_n = nb22.get("ul")
        if dest_r is None or dest_n is None:
            self.skipTest("Geometría no adecuada para oscilación")
        if s["board"][dest_r] != 0 or s["board"][dest_n] != 0:
            self.skipTest("Destino ocupado")

        s["board"][10] =  2
        s["board"][22] = -2

        cur = s
        for _ in range(3):
            cur = step(cur, (10, dest_r))
            cur = step(cur, (22, dest_n))
            cur = step(cur, (dest_r, 10))
            cur = step(cur, (dest_n, 22))

        self.assertTrue(is_terminal(cur), "Debe ser terminal por triple repetición")
        self.assertEqual(result(cur), 0)


class TestGeneral(unittest.TestCase):
    def test_initial_legal_moves(self):
        s = initial_state()
        moves = legal_moves(s)
        self.assertEqual(len(moves), 7)

    def test_step_changes_turn(self):
        s = initial_state()
        s2 = step(s, legal_moves(s)[0])
        self.assertEqual(s2["turn"], -1)

    def test_illegal_move_raises(self):
        s = initial_state()
        with self.assertRaises(ValueError):
            step(s, (0, 31))

    def test_encode_length(self):
        from src.damas.engine import encode
        s = initial_state()
        enc = encode(s)
        self.assertEqual(len(enc), 5 * 32)


if __name__ == "__main__":
    unittest.main(verbosity=2)