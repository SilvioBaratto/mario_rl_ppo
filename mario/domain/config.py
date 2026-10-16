"""Constants that every layer shares: the default level and the facts about the emulator."""

from __future__ import annotations

DEFAULT_ENV_ID = "SuperMarioBros-1-1-v0"

# Two agent steps between reverse-curriculum start states: fine enough that
# each step back asks for only a couple of new decisions, coarse enough that
# the curriculum crosses a ~300-frame route in a few dozen moves.
DEMO_STRIDE = 8
FRAME_SKIP = 4

# SMB1 RAM "float state": 0 while Mario stands on solid ground. Frontier
# snapshots are only taken then, so a restart never begins in a doomed fall.
SMB1_FLOAT_STATE = 0x001D

# Raw screen rows of the SMB1 status bar; the playfield starts at row 32.
HUD_ROWS = 32
