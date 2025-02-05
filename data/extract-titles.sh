#!/bin/sh
egrep -b '<title>' enwiki-20150205-pages-articles.xml  | sed -E 's/<\/?title>//g' | sed 's/: +/,/g' | sed 's/  /,/' > titles.csv
