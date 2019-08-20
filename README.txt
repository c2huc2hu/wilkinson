
In this directory:

- data.txt
  summarizes all scans, transcriptions, etc.
  some scans have inline solutions ("solved"), some are unsolved
    for some of the solved ciphers, we have extracted wordbanks ("banked")
    NOTE: many solved ciphers have not been banked yet!
  for some of the unsolved solutions, we have transcriptions ("transcribed")

- wordbank*
  these are extracted wordbanks

- unsolved.ciphers.1078, unsolved.ciphers.1096 
  these are transcribed, unsolved ciphers
  the "1078" ciphers use one system, the "1096" ciphers use another

- apply-wordbank
  script that applies current wordbank (without 2880) to any unsolved cipher

- unsolved.ciphers.1078.apply  
  we'd like to "fill in the blanks" to complete the solution

- dict.modern
  a dictionary downloaded from the web (only includes root forms, like pocket dict)

==============================================================

Places to go:

Overall spreadsheet:
https://docs.google.com/spreadsheets/d/1poDrS2RNBJjLpv8gLZwAj5NjLU2ok51EgBhNzXYEIek/edit#gid=0

Wordbanking:
https://docs.google.com/spreadsheets/d/1poDrS2RNBJjLpv8gLZwAj5NjLU2ok51EgBhNzXYEIek/edit#gid=1271821898

Transcriptions of unsolved ciphers:
https://docs.google.com/document/d/1PlTOx3_g4K61FbP98I-DSn3r9hh7TbyOPzRiYd88I08/edit

Burr cipher and other stuff:
https://drive.google.com/drive/folders/1Q9uALIZiYlsKi5p8Yn2oIhqKf7lv9NcW

Image files:
https://drive.google.com/drive/folders/1RVv4348MrSfUUIdUqJVuq6Fak1Y9_sJW

Files from Austin Wheeler:
(on PC.  wilkinson/austin/)

==============================================================

Current wordbanks come from these five ciphers:

   word
 tokens cipher
    167 2880
     92 1652
     52 2916
     28 2877
     15 1633

Other solved ciphers include: 2958, 2990, 2998, 3063

==============================================================

Stats on these 5 wordbanks:

Cipher 2880
Table words (^):    74
Dict words (-):     48
Dict words (=):     45
Rows go up to 39        /* different dictionary */

Cipher 1652
Table words (^):    37
Dict words (-):     30
Dict words (=):     24
Rows go up to 29

Cipher 2916
Table words (^):    11
Dict words (-):     22
Dict words (=):     19
Rows go up to 29

Cipher 2877
Table words (^):    11
Dict words (-):     16   /* something off here. */
Dict words (=):     1    /* almost all column 1 */
Rows go up to 29

Cipher 1633
Table words (^):    8
Dict words (-):     5
Dict words (=):     2
Rows go up to 29

Let's drop wordbanks from ciphers 2880 and 2877, and 
join the other three into Wordbank(1652+2916+1633)

==============================================================

Stats on Wordbank(1652+2916+1633)

Table words (^):    56
Dict words (-):     57
Dict words (=):     45

   freq row-number   /* nice, even distribution */
      4 1
      4 2
      4 3
      4 4
      4 5
      4 6
      1 7
      6 8
      5 9
      3 10
      9 11
      6 12
      3 13
      6 14
      1 15
      3 16
      3 17
      3 18
      1 19
      6 20
      4 21
      5 22
      3 24
      2 25
      2 28
      4 29
      1 40  /* we can ignore this outlier */

==============================================================

Dictionary from Wordbank(1652+2916+1633):

[13]^	1652	Wilkinson
[101]^	2916	Baron

[160]^	1652	a
[172]^	1652	and
[231]^	1652	bears
[313]^	1652	by
[341]^	1652	charge
[429]^	1652	days
[458]^	2916	does
[527]^	2916	fine
[545]^	1652	???
[548]^	1652	???
[570]^	1652	from
[570]^	2916	from
[598?]^	1652	God
[628]^	1633	have
[629]^	1652	he
[651]^	1652	had
[653]^	1652	hope
[664]^	1633	i
[676]^	1633	is
[681]^	1652	keep
[723]^	1652	line
[729]^	1633	long
[730]^	1652	look
[752]^	1652	man
[798]^	1652	not
[803]^	1633	of
[814]^	1652	own
[855]^	1652	???
[975]^	1633	since
[994]^	1652	some
[997]^	2916	soon
[1049]^	2916	such
[1077]^	1652	that
[1078]^	1652	the
[1096]^	1652	thing
[1104]^	1652	they
[1105]^	2916	time
[1106]^	1633	to
[1107]^	1652	tool
[1123]^	1652	trust? that?
[1151]^	2916	wait
[1186]^	1652	who
[1191]^	2916	shall
[1191]^	1652	will
[1197]^	1652	with
[1206]^	2916	write
[1216]^	1652	you
[1218]^	2916	your
[1219]^	1652	young
[1233]^	1633	it
[1234]^	1652	him
[1235]^	1652	my 
[1235]^	1652	me
[1245]^	1652	his

007.[24]-	1652	acquisition
009.[14]=	1652	Adieu?
015.[21]-	1652	after
030.[29]-	2916	???
036.[8]^	1652	April
044.[28]-	1652	attache?
047.[21]-	2916	attachment
059.[19]-	1652	bearer
065.[17]=	1652	better
075.[29]-	1633	bosom
075.[29]-	1652	bosom
103.[40]=	2916	Chambers
113.[4]-	2916	cipher
114.[20]-	1652	civility
123.[2]=	2916	dispatch
139.[8]-	1633	confidence
142.[13]=	1652	concerns
151.[2]=	1652	commitment?
152.[5]-	1652	correspondence
179.[25]-	1633	December
189.[9]=	2916	depends
192.[10]=	2916	deserve
192.[22]=	1652	???
216.[14]=	1652	dollars
237.[12]-	1652	enclose
237.[7]=	1652	encouraged
238.[12]-	2916	enclosed
239.[17]-	1652	engaged
250.[1]-	1652	???
250.[11]-	1652	eventful?
251.[6]=	2916	every
256.[3]-	2916	exertions
259.[12]-	2916	express
259.[12]-	2916	???
269.[29]-	2916	favor
284.[6]=	1652	follow
286.[14]=	2916	for
290.[5]-	1652	fortune
293.[20]=	1633	friend
293.[20]=	1652	Friend
305.[3]-	1652	gentleman
337.[25]=	1652	high
360.[4]-	2916	implement
361.[2]-	1652	important
375.[1]=	1652	information
376.[4]-	1652	intrigued
381.[18]-	1652	measurable?
385.[21]=	2916	intelligence
386.[5]-	1652	interesting
399.[22]=	2916	just
401.[20]-	1652	kinship
402.[24]-	2916	King
413.[8]=	1652	letter
448.[21]-	2916	moment
455.[22]=	2916	My
457.[20]-	1652	???
460.[11]-	2916	???
461.[1]-	1652	no
464.[13]-	2916	notice
474.[??]-	1652	or
474.[17]-	1652	or
474.[5]=	2916	opportunity
476.[3]-	2916	orders
483.[14]-	1652	packet
486.[9]-	1652	papers
500.[28]=	2916	permit
510.[16]-	1652	plan
515.[11]=	1652	political
515.[11]=	1652	political
515.[11]=	1652	political
523.[9]=	1652	precursor?
536.[22]-	1652	property
539.[11]-	1652	protection
543.[15]=	1652	pursuit
545.[11]=	2916	put
546.16]-	1652	quarter?
550.[20]=	1652	Personally?
555.[11]-	1652	reception
556.[13]-	2916	receive
556.[9]=	1652	???
557.[1]=	2916	recommends
612.[10]-	2916	???
648.[14]=	1652	strongly
648.[8]-	2916	???
652.[18]-	2916	subject?
652.[18]-	2916	subject
672.[9]-	2916	tell
678.[16]=	2916	This
678.[6]=	2916	this
678.[6]=	2916	this
682.[22]-	2916	timely
689.[3]=	2916	transacted
697.[8]-	1633	truth
720.[4]-	2916	undertaking
724.[12]=	1652	uninformed
739.[8]-	1652	understanding?
743.[12]=	2916	use
743.[8]=	2916	us
754.[24]=	1652	whose
762.[11]=	1633	written
764.[14]-	1633	you
764.[2]=	1652	yourself
859.[10]=	1652	support   /* outlier */

From this, we also see:

- The table (^) is about 1200 lines.
- The dictionary (nnn.nn) is about 750 pages.
- At 29 words per column, that comes to about 43,500 dict words.

==============================================================

Unsolved cipher, with ranges (indented) for unknown words:

[219]^	been
[737]^	made
[1068]^	the
[771]^	UNKNOWN

  [752^]  1652    man
  [798]^  1652    not

[794]^	from
[1068]^	the
[88]^	UNKNOWN

  PROPER NAME

[1096]^	to
[603]^	UNKNOWN

  [598?]^ 1652    God
  [628]^  1633    have

[812]^	UNKNOWN

  [814]^  1652    own
  [975]^  1633    since

[794]^	from
[1068]^	the
223.[20]-	UNKNOWN

  216.[14]=       1652    dollars
  237.[12]-       1652    enclose

[162]^	and
[1096]^	to
254.[5]-	UNKNOWN

  251.[6]=        2916    every
  256.[3]-        2916    exertions

[1212]^	an
178.[14]=	UNKNOWN

  152.[5]-        1652    correspondence
  179.[25]-       1633    December

into	into

[1068]^	the
098.[10]=	UNKNOWN

  075.[29]-       1652    bosom
  103.[40]=       2916    Chambers

       etc etc etc

==============================================================

Let's take an example tight range:

254.[5]-	"...from the ? and to <this word> an..."

  251.[6]=        2916    every
  256.[3]-        2916    exertions

So the unknown word is between "every" & "exertion".  It's 
probably a verb, since it follows "to".

Assuming a 29-row dictionary:
  - There are 258 words between "every" & "exertion"
  - Our target word is the 144th one after "every"
  - 144/258 = 56% of the distance from "every" to "exertion"

Let's take a compact, modern 20,000-word dictionary.
  - There are 92 words between "every" & "exertion"
  - Words near 56% of the distance from "every" to "exertion":

exceedingly
excel
excellence
excellent
excellently
except
exception
exceptional
exceptionally
excerpt                <== 56%
excess
excesses
excessive
excessively
exchange
excise
excision
excitable
excite
excited

So maybe "exchange", "excel", "excite"?

We'll never know, supposing "254.[5]-" isn't in our wordbank.

But we could take another solved cipher and pretend it's not 
solved.  I.e. treat it like the unsolved one above, substituting
only matches from Wordbank(1652+2916+1633).  Then we when
work on its remaining "unknown" words (like "254.[5]-" above), 
we would actually have the right answer to compare our guess to.

====

Proper names in unsolved cipher (codename = "[...]^")

   freq codename [...]^
      5 8
      7 10
      1 16
      3 18
      1 19
      3 20
      1 23
      1 24
      3 28
      1 39
      1 44
      1 53
      1 57
     10 61
      2 72
      2 73
      2 75
      1 76
      2 79
      1 81
      5 83
      4 88
      1 90
      5 101   the "Baron"
      1 103
      1 105
      1 106
      1 108
      1 117
      1 118
      1 120
      2 133

