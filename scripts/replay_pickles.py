#!/usr/bin/env python3
"""Command-line entrypoint for plotting archived experiment data."""
from scripts.reproduce_article import main,reproduce

def replay(article,folder,out,trust_pickle=False,manifest=None,methods=None,
           max_representations=8,include_times=False):
    return reproduce(article,folder,out,trust_pickle)

if __name__=='__main__':main()
