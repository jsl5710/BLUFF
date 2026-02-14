# BLUFF Language Taxonomy

This document provides the complete linguistic classification for all 79 languages in the BLUFF benchmark, organized by resource category, language family, script type, and syntactic word order.

---

## Resource Categories

### Big-Head Languages (20)

High-resource languages with substantial NLP training data and tool support.

| Code | Language | Family | Script | Syntax | Region |
|------|----------|--------|--------|--------|--------|
| ar | Arabic | Afro-Asiatic | Arabic | VSO | Middle East & North Africa |
| bn | Bengali | Indo-European | Bengali | SOV | South Asia |
| de | German | Indo-European | Latin | SVO | Europe |
| en | English | Indo-European | Latin | SVO | Global |
| es | Spanish | Indo-European | Latin | SVO | Europe & Latin America |
| fa | Persian | Indo-European | Arabic | SOV | Middle East |
| fr | French | Indo-European | Latin | SVO | Europe & Africa |
| hi | Hindi | Indo-European | Devanagari | SOV | South Asia |
| id | Indonesian | Austronesian | Latin | SVO | Southeast Asia |
| it | Italian | Indo-European | Latin | SVO | Europe |
| ja | Japanese | Japonic | CJK | SOV | East Asia |
| ko | Korean | Koreanic | CJK | SOV | East Asia |
| nl | Dutch | Indo-European | Latin | SVO | Europe |
| pl | Polish | Indo-European | Latin | SVO | Europe |
| pt | Portuguese | Indo-European | Latin | SVO | Europe & Latin America |
| ru | Russian | Indo-European | Cyrillic | SVO | Europe & Central Asia |
| sv | Swedish | Indo-European | Latin | SVO | Europe |
| tr | Turkish | Turkic | Latin | SOV | Europe & Central Asia |
| uk | Ukrainian | Indo-European | Cyrillic | SVO | Europe |
| zh | Chinese | Sino-Tibetan | CJK | SVO | East Asia |

### Long-Tail Languages (58)

Low-resource languages with limited NLP resources, underrepresented in LLM training data.

| Code | Language | Family | Script | Syntax | Region |
|------|----------|--------|--------|--------|--------|
| af | Afrikaans | Indo-European | Latin | SVO | Southern Africa |
| am | Amharic | Afro-Asiatic | Ethiopic | SOV | East Africa |
| az | Azerbaijani | Turkic | Latin | SOV | Central Asia |
| bg | Bulgarian | Indo-European | Cyrillic | SVO | Europe |
| ca | Catalan | Indo-European | Latin | SVO | Europe |
| ceb | Cebuano | Austronesian | Latin | VSO | Southeast Asia |
| cs | Czech | Indo-European | Latin | SVO | Europe |
| cy | Welsh | Indo-European | Latin | VSO | Europe |
| da | Danish | Indo-European | Latin | SVO | Europe |
| el | Greek | Indo-European | Greek | SVO | Europe |
| et | Estonian | Uralic | Latin | SVO | Europe |
| eu | Basque | Language Isolate | Latin | SOV | Europe |
| fi | Finnish | Uralic | Latin | SVO | Europe |
| ga | Irish | Indo-European | Latin | VSO | Europe |
| gl | Galician | Indo-European | Latin | SVO | Europe |
| gu | Gujarati | Indo-European | Gujarati | SOV | South Asia |
| ha | Hausa | Afro-Asiatic | Latin | SVO | West Africa |
| he | Hebrew | Afro-Asiatic | Hebrew | VSO | Middle East |
| hr | Croatian | Indo-European | Latin | SVO | Europe |
| hu | Hungarian | Uralic | Latin | SVO | Europe |
| hy | Armenian | Indo-European | Armenian | SOV | Caucasus |
| is | Icelandic | Indo-European | Latin | SVO | Europe |
| ka | Georgian | Kartvelian | Georgian | SOV | Caucasus |
| kk | Kazakh | Turkic | Cyrillic | SOV | Central Asia |
| km | Khmer | Austroasiatic | Khmer | SVO | Southeast Asia |
| kn | Kannada | Dravidian | Kannada | SOV | South Asia |
| ku | Kurdish | Indo-European | Latin/Arabic | SOV | Middle East |
| ky | Kyrgyz | Turkic | Cyrillic | SOV | Central Asia |
| lo | Lao | Tai-Kadai | Lao | SVO | Southeast Asia |
| lt | Lithuanian | Indo-European | Latin | SVO | Europe |
| lv | Latvian | Indo-European | Latin | SVO | Europe |
| mk | Macedonian | Indo-European | Cyrillic | SVO | Europe |
| ml | Malayalam | Dravidian | Malayalam | SOV | South Asia |
| mr | Marathi | Indo-European | Devanagari | SOV | South Asia |
| ms | Malay | Austronesian | Latin | SVO | Southeast Asia |
| mt | Maltese | Afro-Asiatic | Latin | VOS | Europe |
| my | Burmese | Sino-Tibetan | Myanmar | SOV | Southeast Asia |
| ne | Nepali | Indo-European | Devanagari | SOV | South Asia |
| no | Norwegian | Indo-European | Latin | SVO | Europe |
| pa | Punjabi | Indo-European | Gurmukhi | SOV | South Asia |
| ps | Pashto | Indo-European | Arabic | SOV | Central/South Asia |
| ro | Romanian | Indo-European | Latin | SVO | Europe |
| si | Sinhala | Indo-European | Sinhala | SOV | South Asia |
| sk | Slovak | Indo-European | Latin | SVO | Europe |
| sl | Slovenian | Indo-European | Latin | SVO | Europe |
| sq | Albanian | Indo-European | Latin | SVO | Europe |
| sr | Serbian | Indo-European | Cyrillic/Latin | SVO | Europe |
| sw | Swahili | Niger-Congo | Latin | SVO | East Africa |
| ta | Tamil | Dravidian | Tamil | SOV | South Asia |
| te | Telugu | Dravidian | Telugu | SOV | South Asia |
| tg | Tajik | Indo-European | Cyrillic | SOV | Central Asia |
| th | Thai | Tai-Kadai | Thai | SVO | Southeast Asia |
| tl | Tagalog | Austronesian | Latin | VSO | Southeast Asia |
| ur | Urdu | Indo-European | Arabic | SOV | South Asia |
| uz | Uzbek | Turkic | Latin | SOV | Central Asia |
| vi | Vietnamese | Austroasiatic | Latin | SVO | Southeast Asia |
| xh | Xhosa | Niger-Congo | Latin | SVO | Southern Africa |
| yo | Yoruba | Niger-Congo | Latin | SVO | West Africa |

---

## Language Family Distribution

| Family | Count | Languages |
|--------|-------|-----------|
| Indo-European | 43 | en, de, fr, es, pt, it, nl, sv, da, no, pl, cs, sk, hr, sr, bg, mk, sl, uk, ru, ro, sq, el, lt, lv, is, ga, cy, ca, gl, af, hi, bn, ur, gu, mr, pa, ne, si, fa, ku, ps, hy, tg |
| Turkic | 5 | tr, az, kk, ky, uz |
| Afro-Asiatic | 5 | ar, he, ha, am, mt |
| Austronesian | 5 | id, ms, tl, ceb, mt |
| Dravidian | 4 | ta, te, kn, ml |
| Uralic | 3 | fi, hu, et |
| Niger-Congo | 3 | sw, yo, xh |
| Sino-Tibetan | 2 | zh, my |
| Tai-Kadai | 2 | th, lo |
| Austroasiatic | 2 | vi, km |
| Japonic | 1 | ja |
| Koreanic | 1 | ko |
| Kartvelian | 1 | ka |
| Language Isolate | 1 | eu |

## Script Type Distribution

| Script | Count | Languages |
|--------|-------|-----------|
| Latin | 43 | en, de, fr, es, pt, it, nl, sv, da, no, pl, cs, sk, hr, sl, ro, sq, lt, lv, is, ga, cy, ca, gl, af, id, ms, tl, ceb, mt, tr, az, uz, sw, yo, xh, ha, fi, hu, et, eu, vi, ku |
| Cyrillic | 8 | ru, uk, bg, mk, sr, kk, ky, tg |
| Arabic | 4 | ar, fa, ur, ps |
| Devanagari | 3 | hi, mr, ne |
| CJK | 3 | zh, ja, ko |
| Dravidian scripts | 4 | ta, te, kn, ml |
| Thai/Lao | 2 | th, lo |
| Bengali | 1 | bn |
| Ethiopic | 1 | am |
| Khmer | 1 | km |
| Georgian | 1 | ka |
| Armenian | 1 | hy |
| Myanmar | 1 | my |
| Other Indic | 3 | gu, pa, si |

## Syntactic Word Order Distribution

| Order | Count | Description | Languages |
|-------|-------|-------------|-----------|
| SVO | 47 | Subject-Verb-Object | en, de, fr, es, pt, it, nl, sv, da, no, pl, cs, sk, hr, sr, bg, mk, sl, uk, ru, ro, sq, el, lt, lv, is, af, ca, gl, id, ms, sw, yo, xh, ha, fi, hu, et, eu, vi, zh, th, lo, km, my + others |
| SOV | 25 | Subject-Object-Verb | hi, bn, ur, gu, mr, pa, ne, si, fa, ku, ps, ta, te, kn, ml, hy, tg, tr, az, kk, ky, uz, ja, ko, ka |
| VSO | 5 | Verb-Subject-Object | ar, he, ga, cy, ceb, tl |
| VOS | 1 | Verb-Object-Subject | mt |

---

## Geographic Coverage (12 Regions)

| Region | Languages |
|--------|-----------|
| Western Europe | en, de, fr, es, pt, it, nl, sv, da, no, is, ga, cy, ca, gl, eu, mt |
| Eastern Europe | pl, cs, sk, hr, sr, bg, mk, sl, ro, sq, el, lt, lv, hu, et, fi |
| Eastern Europe/Central Asia | ru, uk, tg |
| Middle East & North Africa | ar, he, fa, ku, ps, tr |
| South Asia | hi, bn, ur, gu, mr, pa, ne, si, ta, te, kn, ml |
| East Asia | zh, ja, ko |
| Southeast Asia | id, ms, tl, ceb, th, lo, vi, km, my |
| Central Asia | az, kk, ky, uz |
| West Africa | ha, yo |
| East Africa | am, sw |
| Southern Africa | af, xh |
| Caucasus | hy, ka |
