# Constrained adversarial attacks

Experimental setup for introducing constraints to universal adversarial attacks.

### Usage

Install dependencies

```
pip install -r requirements.txt
```

Run all experiments for all configuration options:

```
make all
```

Usage help:

```
python3 -m exp --help
```

### Experiment workflow

<pre>
    __main__.py                          experiment.py
   ┌───────────┐     ┌────────────┐      ┌────────────┐      ┌────────────┐
○──┤   parse   ├─────┤   setup    ├──────┤    run     ├──────┤    end     ├──◎
   │   args    │     │ experiment │      │ experiment │      │ experiment │
   └───────────┘     └────────────┘      └────────────┘      └────────────┘
    from config      * preprocess         k times:           * write result
    & cmd args       * init model         1. train model     * plot graph
                     * init attack        2. attack
                     * init validation    3. score
</pre>


### Development instructions

First install all dev dependencies:

```
pip install -r requirements-dev.txt
```

Available code quality checks

<pre>
make test    -- Run unit tests
make lint    -- Run linter
make dev     -- Test and lint, all at once
</pre>