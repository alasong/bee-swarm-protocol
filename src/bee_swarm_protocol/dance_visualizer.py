"""ASCII visualization of dance positions and intensity heat maps."""

from collections import defaultdict
from typing import Dict, List

from bee_swarm_protocol.dance_propagation import Dance, DancePropagator


class DanceVisualizer:
    """
    Visualizes dances for debugging and monitoring.

    - ASCII grid visualization with intensity-based symbols
    - Heat map by location zone and dance type
    """

    SYMBOLS = [(".", 0.3), ("o", 0.5), ("O", 0.7), ("*", 0.9), ("X", 1.1)]

    def __init__(
        self,
        propagator: DancePropagator,
        grid_width: int = 20,
        grid_height: int = 10,
    ):
        self._propagator = propagator
        self.grid_width = grid_width
        self.grid_height = grid_height
        self._min_x = 0.0
        self._max_x = 500.0
        self._min_y = 0.0
        self._max_y = 500.0

    def visualize(self, dances: List[Dance]) -> str:
        """
        Create ASCII visualization of dances.

        Symbols: .(low) o(medium) O(high) *(very high) X(max)
        """
        grid = [[" " for _ in range(self.grid_width)] for _ in range(self.grid_height)]

        for dance in dances:
            if dance.location is None:
                continue
            gx, gy = self._to_grid_coords(dance.location.x, dance.location.y)
            if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                intensity = (
                    dance.decayed_intensity
                    if dance.decayed_intensity > 0
                    else dance.original_intensity
                )
                grid[gy][gx] = self._get_intensity_symbol(intensity)

        lines = ["+" + "-" * self.grid_width + "+"]
        for row in grid:
            lines.append("|" + "".join(row) + "|")
        lines.append("+" + "-" * self.grid_width + "+")
        lines.append("Symbols: .(low) o(medium) O(high) *(very high) X(max)")
        return "\n".join(lines)

    def get_heat_map(self) -> Dict[str, float]:
        """Get intensity heat map by zone and type."""
        heat_map: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)

        for dance in self._propagator.get_all_active_dances():
            intensity = (
                dance.decayed_intensity
                if dance.decayed_intensity > 0
                else dance.original_intensity
            )
            if dance.location:
                zone = dance.location.zone
                heat_map[f"zone:{zone}"] += intensity
                counts[f"zone:{zone}"] += 1
            heat_map[f"type:{dance.signal.direction}"] += intensity
            counts[f"type:{dance.signal.direction}"] += 1

        for key in heat_map:
            if counts[key] > 0:
                heat_map[key] = heat_map[key] / counts[key]
        return dict(heat_map)

    def _to_grid_coords(self, x: float, y: float) -> tuple:
        gx = int((x - self._min_x) / (self._max_x - self._min_x) * self.grid_width)
        gy = int((y - self._min_y) / (self._max_y - self._min_y) * self.grid_height)
        return (
            max(0, min(self.grid_width - 1, gx)),
            max(0, min(self.grid_height - 1, gy)),
        )

    def _get_intensity_symbol(self, intensity: float) -> str:
        for symbol, threshold in self.SYMBOLS:
            if intensity < threshold:
                return symbol
        return "X"

    def set_bounds(self, min_x: float, max_x: float, min_y: float, max_y: float) -> None:
        """Set spatial bounds for visualization."""
        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y
