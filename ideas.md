# Audio Track Separation

Moises in a day?

Tbd what models are available.

## Training approach:

- start with actual stems.
- create mixes of those stems
- put mixes in an input directory
- put stems in an output directory
- train the model until it can recreate 
  the source stems given a mix



## Training data:

- start with what I have
  - look at my Ardour projects
- search for open source stems
  - what licenses are suitable?
- Make a few at home?
  - guitar / bass / vocal / percussion jam
  - add drum loops
  - add synth?
  - use open source stems for instr I don't have
    and play along w them
- purchase Rock Band games or equivalent 
  - script the extraction of stems

## Creating mixes

Automate this with Ecasound or similar
Choose a scriptable engine with these capabilities.

The mixes probably don't have to be GOOD

But maybe they need to do things real mixes do

Start simply and iterate:

Simplest possible:

- combine stems at equal volume

Progressively add features and parameters:

- Track level
- Track pan

Per-stem effects (add to stem, these
should be expected as part of stem output)

- Track EQ
- track compression
- track effects
- track muting

Mix effects

- reverb/delay buses, possibly multiple
- mix bus compression
- for these we probably want to add to stems
  by recording a mix with other stems muted

Automation: evolve parameters during mix

## Data Augmentation 

We can do the equivalent of Random Resized Crop
by:
- resampling so pitch/time changes by x %
- choosing an arbitrary section if desired length

Just apply that to both the stems and the mix.

We can create N different mixes from a set of
stems by
randomizing N sets of parameters




# Music improv partner agent

## First: can i find/fine-tune/train a model with pitch recognition?

can i get good control data prediction (maybe OSC) out of
- a sound file?
- a realtime audio stream??

If I can't do realtime this project is dead

*If this is solved already I'd rather use something that just works than build it*
though it might be fun to timebox a minimal proof of concept

what's on huggingface?
what exists that's not AI?


challenges:
- inexact tuning
- bent notes
- vibrato
- dynamics
- polyphony!!
- latency: time to output

## next: can i train a model to predict future events for some time?

input/output could be OSC

store what i've played before as context?
and what the model has generated?
previous streams of note event data
(how do we model this as context)

this could be _kind of_ seen as a chatbot:
- here's the musical context up through "now", keep it going
- what's the "return key" equivalent ???
  - some kind of rolling window?

How do we connect that to a synthesizer?
Maybe a separate non-AI program receives the bot's generated OSC
and knows how to weave that into a continuous stream it's
sending to an OSC-controlled synth

Context window: maybe rolling (forget earlier stuff)?


### step 1 goal: continue my monophonic line? (looper?)

maybe don't get bogged down here. proof of concept

### step 2-N goal: invent accompaniment!


Possibly simple but effective breakdown:
- follow last instruction until told otherwise
- 3 "bands" of activity (bass, mid, high) - or "players"
- each is monophonic
- meta control signals i'd want to be able to send:
  - in band X, make more room for me / stop
  - in band X, take over from me with more/less variation
  - in band X, fight me
  - in band X, get busier/simplify
  - in band X, get more/less orderly/chaotic
  - all of the above but you're in band X and i'm in band Y
  - in band X, invent accompaniment for bands Y and Z
    - busy / sparse
    - harmonic / contrapuntal


Rhythm recognition:
"easy": loose, ambient, sparse
"hard": rhythmic, metered, syncopated, variations

### challenge: does this work at all?

unknown!

### challenge: WHAT IS THE TRAINING DATA

public domain music?
then i jam at it a while?
how much do we need?
will it sound generic / lame?

### challenge: latency

how high is too high?
Quick response to my playing and commands

### side quest: what's the synthesizer?
i don't want to have to spend a lot of time programming/tweaking synths
- easiest would be to do some osc equivalent of "general midi"
- or describe to a model what i want and it makes the synths???

### side quest: sampling??

how hard is realtime sampling instead of / in addition to the above

- could it grab "notes" and use those for playback
  - time / pitch stretching
  - could it do interesting things with phrases



## next add drums

how much do i want to control it
