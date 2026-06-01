# Weather Stem Selector MVP

Layer B MVP handler for short weather-only stems.

Frontend controls:

- weather type: `rain`, `wind`, `thunder`, `storm`
- intensity: `light`, `medium`, `heavy`
- duration in seconds
- seed

The seed controls both asset selection and start offset inside longer assets.
Layer B returns a short WAV stem plus metadata; Layer D owns final timeline
placement and full soundscape mixing.

