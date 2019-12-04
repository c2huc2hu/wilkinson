Scraped a bunch of sentences from COFEA by searching for 100k sentences from founders online with "A" in them
https://lcl.byu.edu/projects/cofea/

Filter out sentences in French and Spanish with super naive filters:
```
.*[àáâèéêëîïôûçñ].*\n -> 
.*\b(un|une|alors|avec|vous)\b.*\n ->
& -> and

```

Yields 21M characters, which aren't that high quality