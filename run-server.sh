#!/bin/bash

python -Xgil=0 -m parallelopedia.http.server                                  \
    --ip 0.0.0.0                                                              \
    --port 4444                                                               \
    --threads $(nproc)                                                        \
    --log-level INFO                                                          \
    --app-classes parallelopedia.http.server.PlaintextApp                     \
                  parallelopedia.gpt2.Gpt2App                                 \
                  parallelopedia.wiki.WikiApp                                 \
                  parallelopedia.llm.CausalModelApp

