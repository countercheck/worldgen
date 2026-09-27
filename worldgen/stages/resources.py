"""Settlements that exist for something other than the farmland around them.

A market is where the countryside sells its surplus, and `MarketStage` sites every one of
them on that alone. Three kinds of place are not like that, and no amount of fertility
will put one where it belongs:

**Ports** stand where cargo changes hands. `CityPromotionStage` routes every city's
provisioning home and pays each quay on the way a share of what it handles; where no
settlement stands within reach of a busy quay, that share stayed with the city. A port is
founded there and takes it.

**Mines** stand on the ore, which is in the hills — on exactly the ground a market would
never choose — but only where the ore can get out. Worked ore was smelted at the mine and
the metal carried to the nearest river port or city, further than grain would go because a
ton of it was worth more (Pennine lead went overland to river ports such as Bawtry), but
not without limit.

**Lumber camps** stand in the big woods, on a river that will float the timber out, and
only where the timber can reach a city that wants it. Hauling timber overland was slow
and ruinous; floating it was the whole trade.

None of them grows its own food, which is the point: like a city, each is a place fed from
beyond itself. So none of them conjures people. A new workforce is drawn from the
settlements that can haul food to it — nearer ones sending more, and none more than
`resource_draw_share` of what it could — and the map's total population is unchanged.
"""

from ..core.hex import LandUse, Settlement, SettlementRole, SettlementTier
from ..core.hex_grid import distance, grade_reachable_count, hex_range, neighbors
from ..core.pipeline import GeneratorStage
from ..core.world_state import WorldState
from .haulage import bulk_routes, floatable, usable_fraction
from .road_cost import WATER, grade_is_under_cap


class ResourceStage(GeneratorStage):
    """Founds ports at unattended quays, mines on the ore, and lumber camps in the woods."""

    def run(self, state: WorldState) -> WorldState:
        hexes = state.hexes
        cfg = self.config

        reach_cache: dict = {}

        def reachable(coord) -> bool:
            """The same test every settlement passes: enough ground a cart can get about."""
            if coord not in reach_cache:
                reach_cache[coord] = grade_reachable_count(
                    coord,
                    hexes,
                    lambda a, b: grade_is_under_cap(a, b, cfg),
                    cfg.settlement_min_reachable,
                )
            return reach_cache[coord] >= cfg.settlement_min_reachable

        self._ports(state, reachable)
        worked = self._mines(state, reachable) + self._lumber(state, reachable)
        self._ship(state, worked)
        return state

    # -- ports ----------------------------------------------------------------

    def _ports(self, state, reachable) -> None:
        """A port on the busiest unattended quays, paid what the cities were keeping for them.

        Quays within `transship_radius` of each other are one waterfront, so they are pooled
        onto the busiest of them before the threshold is tested.
        """
        cfg = self.config
        hexes = state.hexes
        rows = state.metadata.pop("unhandled_quays", [])
        if not rows or cfg.port_min_population <= 0:
            return

        by_quay: dict = {}
        for q, r, sq, sr, food, people in rows:
            entry = by_quay.setdefault((q, r), {"food": 0.0, "from": {}})
            entry["food"] += food
            entry["from"][(sq, sr)] = entry["from"].get((sq, sr), 0.0) + people

        by_coord = {s.coord: s for s in state.settlements}
        founded = 0
        taken: set = set()
        for quay in sorted(by_quay, key=lambda c: (-by_quay[c]["food"], c)):
            if quay in taken:
                continue
            front = [
                c for c in hex_range(quay, cfg.transship_radius) if c in by_quay and c not in taken
            ]
            owed: dict = {}
            for c in front:
                for seat, people in by_quay[c]["from"].items():
                    owed[seat] = owed.get(seat, 0.0) + people
            taken.update(front)
            if sum(owed.values()) < cfg.port_min_population:
                continue
            hx = hexes[quay]
            if hx.settlement is not None or hx.terrain_class in WATER or not reachable(quay):
                continue

            # Work out every transfer before making any, so a port that falls short takes
            # nobody with it.
            transfers = [
                (by_coord[seat], min(round(people), by_coord[seat].population - 1))
                for seat, people in sorted(owed.items())
                if seat in by_coord
            ]
            moved = sum(n for _, n in transfers if n > 0)
            if moved < cfg.port_min_population:
                continue
            for city, n in transfers:
                if n > 0:
                    city.population -= n
            # A port that the trade alone makes a city is the entrepôt, and it is one on the
            # same terms as a town grown past `city_min_population`.
            big = 0 < cfg.city_min_population <= moved
            tier = SettlementTier.CITY if big else SettlementTier.TOWN
            self._found(state, quay, tier, SettlementRole.PORT, moved, "port", founded)
            if big:
                state.metadata["cities"] = sorted([*state.metadata.get("cities", []), quay])
            founded += 1

    # -- mines ----------------------------------------------------------------

    def _mines(self, state, reachable) -> list:
        """Ore deposits in high ground, each worked by a village or by the town beside it.

        Deposits are drawn at random over hexes with at least `ore_min_relief_m` of relief,
        weighted towards the higher, at `ore_deposits_per_1000_km2` of that ground and never
        closer than `ore_min_separation`. A deposit is worked only if its metal can reach an
        outlet — a city, or a settlement on water a boat can load at — within
        `haulage_range_land` times `ore_haul_range_mult` by bulk haulage. The size of the
        workforce is drawn around `mine_workforce`, since one seam is not like another.
        """
        cfg = self.config
        hexes = state.hexes
        if cfg.ore_deposits_per_1000_km2 <= 0:
            return []
        outlets = [
            s.coord
            for s in state.settlements
            if s.tier is SettlementTier.CITY or s.role is SettlementRole.PORT
        ]
        if not outlets:
            return []
        to_outlet, _ = bulk_routes(
            hexes, outlets, cfg, budget=cfg.haulage_range_land * cfg.ore_haul_range_mult
        )

        upland = sorted(
            c
            for c, hx in hexes.items()
            if hx.terrain_class not in WATER and hx.relief >= cfg.ore_min_relief_m
        )
        if not upland:
            return []
        wanted = int(self.rng.poisson(len(upland) / 1000.0 * cfg.ore_deposits_per_1000_km2))
        weights = [hexes[c].relief for c in upland]
        total = sum(weights)
        order = self.rng.choice(
            len(upland), size=len(upland), replace=False, p=[w / total for w in weights]
        )

        # The deposits are a fact about the ground, laid down first; whether one is worked
        # is a separate question. Filtering while drawing would only move the mines to
        # workable ground and keep their number, when an ore field with no way out should
        # simply go unworked.
        deposits: list = []
        for i in order:
            if len(deposits) >= wanted:
                break
            coord = upland[int(i)]
            if all(distance(coord, d) >= cfg.ore_min_separation for d in deposits):
                deposits.append(coord)

        # One workforce draw per deposit, worked or not, so whether one seam can get its
        # ore out never changes the size of the next.
        sizes = self.rng.lognormal(0.0, cfg.mine_workforce_sigma, size=len(deposits))
        worked = 0
        output: list = []
        for coord, size in zip(deposits, sizes, strict=True):
            if coord not in to_outlet or not reachable(coord):
                continue
            need = cfg.mine_workforce * float(size)
            done = self._work(state, coord, need, SettlementRole.MINING, "mine", worked)
            if done is not None:
                output.append(done)
            worked += 1
        return output

    # -- lumber ---------------------------------------------------------------

    def _lumber(self, state, reachable) -> list:
        """Camps in the big woods, on water that floats the timber out towards a city.

        A camp's worth is the woodland within `lumber_radius`, discounted by how far the
        timber has to go to reach any city by bulk haulage — timber is bulky, and it is worth
        nothing past `haulage_range_land` for the same reason grain is. Camps are placed
        best-first, `lumber_min_separation` apart, while that worth stays above
        `lumber_min_score`.
        """
        cfg = self.config
        hexes = state.hexes
        cities = [s.coord for s in state.settlements if s.tier is SettlementTier.CITY]
        if not cities or cfg.lumber_min_score <= 0:
            return []
        to_city, _ = bulk_routes(hexes, cities, cfg)

        wood = {c for c, hx in hexes.items() if hx.land_use is LandUse.WOOD}

        def floats(coord) -> bool:
            return floatable(hexes[coord], cfg) or any(
                n in hexes and floatable(hexes[n], cfg) for n in neighbors(coord)
            )

        scored = []
        for coord in sorted(wood):
            if coord not in to_city or hexes[coord].settlement is not None or not floats(coord):
                continue
            mass = sum(1 for c in hex_range(coord, cfg.lumber_radius) if c in wood)
            worth = mass * usable_fraction(to_city[coord], cfg.haulage_range_land)
            if worth >= cfg.lumber_min_score:
                scored.append((-worth, coord, mass))
        scored.sort()

        camps: list = []
        output: list = []
        for _, coord, mass in scored:
            if any(distance(coord, c) < cfg.lumber_min_separation for c in camps):
                continue
            if not reachable(coord):
                continue
            camps.append(coord)
            need = mass * cfg.lumber_people_per_wood_hex
            done = self._work(state, coord, need, SettlementRole.LUMBER, "lumber", len(camps) - 1)
            if done is not None:
                output.append(done)
        return output

    # -- where the output goes ------------------------------------------------

    def _ship(self, state, worked) -> None:
        """Record where each mine's ore and each camp's timber goes, for the roads to carry.

        Raw goods go where they fetch most after carriage, as grain does: split between the
        cities in reach by size times the share that survives the haul, raised to
        `city_pull_sharpness`. Ore is carried `ore_haul_range_mult` times as far as grain;
        timber as far as grain. Volume is the workforce, in the same people-it-feeds units
        as every other freight flow. Nothing moves population here — the workforce was
        already drawn — so this is freight for `InterurbanRoadStage` and no more.
        """
        cfg = self.config
        if not worked:
            return
        hexes = state.hexes
        cities = sorted(
            (s.coord, s.population) for s in state.settlements if s.tier is SettlementTier.CITY
        )
        if not cities:
            return
        widest = cfg.haulage_range_land * max(1.0, cfg.ore_haul_range_mult)
        # One search per city, read by every mine and camp: cost of hauling there.
        to = {c: bulk_routes(hexes, [c], cfg, budget=widest)[0] for c, _ in cities}
        freight = state.metadata.setdefault("freight", [])
        for origin, people, role in worked:
            ore = role is SettlementRole.MINING
            reach = cfg.haulage_range_land * (cfg.ore_haul_range_mult if ore else 1.0)
            pull = {}
            for c, pop in cities:
                cost = to[c].get(origin)
                if c != origin and cost is not None and cost < reach:
                    pull[c] = (pop * usable_fraction(cost, reach)) ** cfg.city_pull_sharpness
            total = sum(pull.values())
            for c, p in sorted(pull.items()):
                if p > 0.0:
                    volume = people * p / total
                    freight.append([*origin, *c, round(volume, 3), "ore" if ore else "timber"])

    # -- feeding a workforce --------------------------------------------------

    def _work(self, state, coord, need, role, kind, index):
        """Put *need* people to work at *coord*, fed by whoever can haul food there.

        The town next door takes them on if there is one within `resource_attach_radius` —
        a market beside a seam is a mining town, not a market with a mining village in its
        yard. Otherwise a village is founded, as long as the food that can reach it supports
        at least `resource_min_population`.
        """
        cfg = self.config
        near = [
            (distance(coord, s.coord), s.coord, s)
            for s in state.settlements
            if distance(coord, s.coord) <= cfg.resource_attach_radius
        ]
        host = min(near, key=lambda x: (x[0], x[1]))[2] if near else None
        site = host.coord if host else coord

        people, senders = self._draw(state, site, need, exclude=host)
        if host is not None:
            host.population += people
            return (host.coord, people, role)
        if people < cfg.resource_min_population:
            # Too little food reaches it: give the people back rather than found a hamlet
            # nobody could feed.
            for s, n in senders:
                s.population += n
            return
        self._found(state, coord, SettlementTier.VILLAGE, role, people, kind, index)
        return (coord, people, role)

    def _draw(self, state, site, need, exclude=None) -> tuple[int, list]:
        """Move up to *need* people to *site* from the settlements that can feed it.

        Each sender's weight is its population times the share of a cargo that survives the
        haul (`usable_fraction` over the bulk cost to *site*), and none gives more than
        `resource_draw_share` of that weight. Whole people, so the books balance exactly.
        Returns how many moved, and who sent how many, so a failed founding can undo it.
        """
        cfg = self.config
        cost, _ = bulk_routes(state.hexes, [site], cfg)
        senders = [
            (s, s.population * usable_fraction(cost[s.coord], cfg.haulage_range_land))
            for s in state.settlements
            if s is not exclude and s.coord in cost and s.coord != site
        ]
        senders = [(s, w) for s, w in senders if w > 0.0]
        weight = sum(w for _, w in senders)
        sent: list = []
        if weight <= 0.0:
            return 0, sent
        want = min(need, weight * cfg.resource_draw_share)
        moved = 0
        for s, w in sorted(senders, key=lambda x: x[0].coord):
            n = min(round(want * w / weight), s.population - 1)
            if n <= 0:
                continue
            s.population -= n
            moved += n
            sent.append((s, n))
        return moved, sent

    def _found(self, state, coord, tier, role, population, kind, index) -> None:
        hx = state.hexes[coord]
        biome = hx.biome.name.lower() if hx.biome is not None else "land"
        s = Settlement(
            coord=coord,
            tier=tier,
            role=role,
            population=population,
            name=f"{biome}_{kind}_{index}",
        )
        hx.settlement = s
        state.settlements.append(s)


__all__ = ["ResourceStage"]
