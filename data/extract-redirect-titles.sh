#!/bin/sh
egrep -b '<redirect title' enwiki-20150205-pages-articles.xml  | sed -E 's/<\/?title>//g' | sed 's/: +/,/g' | sed 's/  /,/' > redirect-titles.csv
