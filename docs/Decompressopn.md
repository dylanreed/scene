{\rtf1\ansi\ansicpg1252\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fswiss\fcharset0 Helvetica-Bold;\f2\fmodern\fcharset0 Courier;
\f3\fswiss\fcharset0 Helvetica-Oblique;}
{\colortbl;\red255\green255\blue255;\red255\green255\blue255;\red0\green0\blue0;\red246\green247\blue249;
\red229\green231\blue236;\red154\green154\blue154;\red193\green193\blue193;\red0\green19\blue169;}
{\*\expandedcolortbl;;\cssrgb\c100000\c100000\c100000;\cssrgb\c0\c0\c0;\cssrgb\c97255\c97647\c98039;
\cssrgb\c91765\c92549\c94118;\cssrgb\c66667\c66667\c66667;\cssrgb\c80000\c80000\c80000;\cssrgb\c0\c16863\c72157;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs38 \cf0 \cb2 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec3 Resource files
\fs25\fsmilli12700 \
\pard\pardeftab720\sa400\partightenfactor0

\f1\b \cf0 Table of Contents
\f0\b0 \'a0SCI0 resources SCI1 resources Decompression algorithms\
with significant contributions from Petr Vyhnak and Vladimir Gneushev\
In order to allow games to be both distributeable and playable from several floppy disks, SCI was designed to support multi-volume data. The data itself could therefore be spread into separate files, with some of the more common resources present in more than one of them. The global index for these files was a "resource.map" file, which was read during startup and present on the same disk as the interpreter itself. This file contained a linear lookup table that mapped resource type/number tuples to a set of resource number/ offset tuples, which they could subsequently be read from.\
\pard\pardeftab720\partightenfactor0

\fs38 \cf0 SCI0 resources
\f1\b\fs32\fsmilli16256 \cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf0 \cb2 SCI0 resource.map
\f0\b0\fs25\fsmilli12700 \
\pard\pardeftab720\sa400\partightenfactor0
\cf0 The SCI0 map file format is pretty simple: It consists of 6-byte entries, terminated by the sequence 0xffff ffff ffff. The first 2 bytes, interpreted as little endian 16 bit integer, encode resource type (high 5 bits) and number (low 11 bits). The next 4 bytes are a 32 bit LE integer that contains the resource file number in the high 6 bits, and the absolute offset within the file in the low 26 bits. SCI0 performs a linear search to find the resource; however, multiple entries may match the search, since resources may be present more than once (the inverse mapping is not injective).\
\pard\pardeftab720\partightenfactor0

\f1\b\fs32\fsmilli16256 \cf0 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb2 SCI0 resource.<nr>
\f0\b0\fs25\fsmilli12700 \
\pard\pardeftab720\sa400\partightenfactor0
\cf0 SCI0 resource entries start with a four-tuple of little endian 16 bit words, which we will call (id, comp_size, decomp_size, method). id has the usual SCI0 semantics (high 5 are the resource type, low 11 are its number). comp_size and decomp_size are the size of the compressed and the decompressed resource, respectively. The compressed size actually starts counting at the record position of decomp_size, so it counts four bytes in addition to the actual content. method, finally, is the compression method used to store the data.\
\pard\pardeftab720\partightenfactor0

\fs38 \cf0 SCI1 resources
\f1\b\fs32\fsmilli16256 SCI1 resource.map
\f0\b0\fs25\fsmilli12700 \
\pard\pardeftab720\sa400\partightenfactor0
\cf0 The SCI1 resource.map starts with an array of 3-byte structures where the 1st byte is the resource type (0x80 ... 0x91) and next 2 bytes (interpreted as little-endian 16 bit integer) represent the absolute offset of the resource's lookup table (within resource.map). This first array is terminated by a 3-byte entry with has 0xFF as a type and the offset pointing to the first byte after the last resource type's lookup table. SCI1 first goes through this list to find the start of list for the correct resource type and remember this offset and the offset from the next entry to know where it ends. The resulting interval contains a sorted list of 6-byte structures, where the first LE 16 bit integer is the resource number, and the next LE 32 bit integer contains the resource file number in its high 4 bits and the absolute resource offset (in the indicated resource file) in its low 28 bits. Because the list is sorted and its length is known, Sierra SCI can use binary search to locate the resource ID it is looking for.\
\pard\pardeftab720\partightenfactor0

\f1\b\fs32\fsmilli16256 \cf0 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb2 SCI1 resource.<nr>
\f0\b0\fs25\fsmilli12700 \
\pard\pardeftab720\sa400\partightenfactor0
\cf0 Later versions of SCI1 changed the resource file structure slightly: The resource header now contains a byte describing the resource's type, and a four-tuple (
\f2 \cb4 res_nr, comp_size, decomp_size, method
\f0 \cb2 ), where\'a0
\f2 \cb4 comp_size
\f0 \cb2 ,\'a0
\f2 \cb4 decomp_size
\f0 \cb2 , and\'a0
\f2 \cb4 method
\f0 \cb2 \'a0have the same meanings as before (with the exception of\'a0
\f2 \cb4 method
\f0 \cb2 \'a0referring to different algorithms), while res_nr is simply the resource's number.\
\pard\pardeftab720\partightenfactor0

\fs38 \cf0 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb2 Decompression algorithms
\fs25\fsmilli12700 \
\pard\pardeftab720\sa400\partightenfactor0
\cf0 The decompression algorithms used in SCI are as follows:\
\pard\pardeftab720\partightenfactor0
\cf0 \
\pard\pardeftab720\sa400\partightenfactor0

\f1\b \cf0 Table 2-1. SCI0 compression algorithms
\f0\b0 \

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrt\brdrs\brdrw20\brdrcf6 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clcbpat5 \clwWidth917\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clcbpat5 \clwWidth1666\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\qc\partightenfactor0

\f1\b \cf0 \cb1 method\cell 
\pard\intbl\itap1\pardeftab720\qc\partightenfactor0
\cf0 algorithm\cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth917\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1666\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0

\f0\b0 \cf0 0\cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 uncompressed\cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth917\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1666\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 1\cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 LZW\cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrb\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth917\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1666\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 2\cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 HUFFMAN\cell \lastrow\row
\pard\pardeftab720\partightenfactor0
\cf0 \cb2 \
\pard\pardeftab720\sa400\partightenfactor0

\f1\b \cf0 Table 2-2. SCI01 compression algorithms
\f0\b0 \

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrt\brdrs\brdrw20\brdrcf6 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clcbpat5 \clwWidth917\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clcbpat5 \clwWidth1666\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\qc\partightenfactor0

\f1\b \cf0 \cb1 method\cell 
\pard\intbl\itap1\pardeftab720\qc\partightenfactor0
\cf0 algorithm\cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth917\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1666\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0

\f0\b0 \cf0 0\cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 uncompressed\cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth917\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1666\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 1\cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 LZW\cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth917\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1666\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 2\cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 COMP3\cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrb\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth917\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1666\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 3\cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 HUFFMAN\cell \lastrow\row
\pard\pardeftab720\partightenfactor0
\cf0 \cb2 \
\pard\pardeftab720\sa400\partightenfactor0

\f1\b \cf0 Table 2-3. SCI1.0 compression algorithms
\f0\b0 \

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrt\brdrs\brdrw20\brdrcf6 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clcbpat5 \clwWidth917\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clcbpat5 \clwWidth1666\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\qc\partightenfactor0

\f1\b \cf0 \cb1 method\cell 
\pard\intbl\itap1\pardeftab720\qc\partightenfactor0
\cf0 algorithm\cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth917\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1666\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0

\f0\b0 \cf0 0\cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 uncompressed\cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth917\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1666\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 1\cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 LZW\cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth917\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1666\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 2\cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 COMP3\cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth917\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1666\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 3\cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 UNKNOWN-0\cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrb\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth917\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1666\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 4\cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 UNKNOWN-1\cell \lastrow\row
\pard\pardeftab720\sa400\partightenfactor0

\f1\b \cf0 \cb2 Table 2-4. SCI1.1 compression algorithms
\f0\b0 \cb1 \

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrt\brdrs\brdrw20\brdrcf6 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clcbpat5 \clwWidth917\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clcbpat5 \clwWidth1792\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\qc\partightenfactor0

\f1\b \cf0 \cb2 method\cb1 \cell 
\pard\intbl\itap1\pardeftab720\qc\partightenfactor0
\cf0 \cb2 algorithm\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth917\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1792\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0

\f0\b0 \cf0 \cb2 0\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 uncompressed\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth917\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1792\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 18\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 DCL-EXPLODE\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth917\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1792\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 19\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 DCL-EXPLODE\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrb\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth917\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1792\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 20\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 DCL-EXPLODE\cb1 \cell \lastrow\row
\pard\pardeftab720\sa400\partightenfactor0
\cf0 \cb2 As reported by Vladimir Gneushev, SCI32 uses STACpack (as described in RFC 1974) explicitly, determining whether there is a need for compression by comparing the size of the compressed data block with that of the uncompressed.\cb1 \
\pard\pardeftab720\partightenfactor0

\f1\b\fs32\fsmilli16256 \cf0 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb2 Decompression algorithm LZW
\f0\b0\fs25\fsmilli12700 \cb1 \
\pard\pardeftab720\sa400\partightenfactor0
\cf0 \cb2 The LZW algorithm itself, when used for compression or decompression in an apparatus (sic) designed for compression and decompression, has been patented by Unisys in Japan, Europe, and the United States. Fortunately, FreeSCI only needs LZW decompression, which means that it does not match the description of the apparatus as given above. (Further, patents on software are (at the time of this writing) not enforceable in Europe, where the FreeSCI implementation of the LZW decompressor was written).\cb1 \
\cb2 WriteMe.\cb1 \
\pard\pardeftab720\partightenfactor0

\f1\b\fs32\fsmilli16256 \cf0 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb2 Decompression algorithm HUFFMAN
\f0\b0\fs25\fsmilli12700 \cb1 \
\pard\pardeftab720\sa400\partightenfactor0
\cf0 \cb2 This is an implementation of a simple huffman token decoder, which looks up tokens in a huffman tree.\'a0
\f3\i A huffman tree
\f0\i0 \'a0is a hollow binary search tree. This means that all inner nodes, usually including the root, are empty, and have two siblings. The tree's leaves contain the actual information.\cb1 \
\pard\pardeftab720\partightenfactor0

\f1\b \cf0 \cb7 C Code:\
\pard\pardeftab720\partightenfactor0

\f2\b0 \cf0 \cb4 FUNCTION get_next_bit(): Boolean;\
/* This function reads the next bit from the input stream. Reading starts at the MSB. */\
\
\
FUNCTION get_next_byte(): Byte\
VAR\
    i: Integer;\
    literal: Byte;\
BEGIN\
    literal\'a0:= 0;\
    FOR i\'a0:= 0 to 7 DO\
        literal\'a0:= (literal &#60;&#60; 1) | get_next_bit();\
    OD\
    RETURN literal;\
END\
\
\
FUNCTION get_next_char(nodelist\'a0: Array of Nodes, index\'a0: Integer): (Char, Boolean)\
VAR\
    left, right: Integer;\
    literal\'a0: Char;\
    node\'a0: Node;\
BEGIN\
    Node\'a0:= nodelist[index];\
\
    IF node.siblings == 0 THEN\
	RETURN (node.value, False);\
    ELSE BEGIN\
       left\'a0:= (node.siblings &#38; 0xf0) &#62;&#62; 4;\
       right\'a0:= (node.siblings &#38; 0x0f);\
\
       IF get_next_bit() THEN BEGIN\
	   IF right == 0 THEN /* Literal token */\
	       literal\'a0:= ByteToChar(get_next_byte());\
\
	       RETURN (literal, True);\
	   ELSE\
	       RETURN get_next_char(nodelist, index + right)\
        END ELSE\
	        RETURN get_next_char(nodelist, index + left)\
    END\
END\
\pard\pardeftab720\sa400\partightenfactor0

\f0 \cf0 \cb1 \uc0\u8232 \cb2 The function get_next_char() is executed until its second return value is True (i.e. if a value was read directly from the input stream) while the first return value equals a certain terminator character, which is the first byte stored in the compressed resource:\cb1 \
\pard\pardeftab720\partightenfactor0
\cf0 \

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrt\brdrs\brdrw20\brdrcf6 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clcbpat5 \clwWidth1363\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx2880
\clvertalc \clcbpat5 \clwWidth1991\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx5760
\clvertalc \clcbpat5 \clwWidth3516\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\qc\partightenfactor0

\f1\b \cf0 \cb2 Offset\cb1 \cell 
\pard\intbl\itap1\pardeftab720\qc\partightenfactor0
\cf0 \cb2 Name\cb1 \cell 
\pard\intbl\itap1\pardeftab720\qc\partightenfactor0
\cf0 \cb2 Meaning\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1363\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx2880
\clvertalc \clshdrawnil \clwWidth1991\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx5760
\clvertalc \clshdrawnil \clwWidth3516\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0

\f0\b0 \cf0 \cb2 0\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 terminator\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 Terminator character\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1363\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx2880
\clvertalc \clshdrawnil \clwWidth1991\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx5760
\clvertalc \clshdrawnil \clwWidth3516\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 nodes\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 Number of nodes\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1363\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx2880
\clvertalc \clshdrawnil \clwWidth1991\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx5760
\clvertalc \clshdrawnil \clwWidth3516\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 2 + i*2\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 nodelist[i].value\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 Value of node #i (0 \uc0\u8804  i < nodes)\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1363\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx2880
\clvertalc \clshdrawnil \clwWidth1991\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx5760
\clvertalc \clshdrawnil \clwWidth3516\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 3 + i*2\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 nodelist[i].siblings\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 Sibling nodes of node #i\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrb\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1363\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx2880
\clvertalc \clshdrawnil \clwWidth1991\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx5760
\clvertalc \clshdrawnil \clwWidth3516\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 2 + nodes*2\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 data[]\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 The actual compressed data\cb1 \cell \lastrow\row
\pard\pardeftab720\sa400\partightenfactor0
\cf0 \cb2 where nodelist[0] is the root node.\cb1 \
\pard\pardeftab720\partightenfactor0

\f1\b\fs32\fsmilli16256 \cf0 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb2 Decompression algorithm COMP3
\f0\b0\fs25\fsmilli12700 \cb1 \
\pard\pardeftab720\sa400\partightenfactor0
\cf0 \cb2 WriteMe.\cb1 \
\pard\pardeftab720\partightenfactor0

\f1\b\fs32\fsmilli16256 \cf0 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb2 Decompression algorithm DCL-EXPLODE
\f0\b0\fs25\fsmilli12700 \cb1 \
\pard\pardeftab720\sa400\partightenfactor0
\cf0 \cb2 originally by Petr Vyhnak\cb1 \
\pard\pardeftab720\sa400\partightenfactor0

\f1\b \cf0 \cb2 This algorithm matches one or more of the UNKNOWN algorithms.
\f0\b0 \cb1 \
\pard\pardeftab720\sa400\partightenfactor0
\cf0 \cb2 This algorithm is based on the Deflate algorithm described in the Internet RFC 1951 (see also RFC 1950 for related material). The algorithm is quite similar to the explode algorithm (ZIP method #6 - implode ) but there are differences.\cb1 \
\pard\pardeftab720\partightenfactor0

\f1\b \cf0 \cb7 C Code:\
\pard\pardeftab720\partightenfactor0

\f2\b0 \cf0 \cb4 	/* The first 2 bytes are parameters */\
\
P1 = ReadByte(); /* 0 or 1 */\
	/* I think this means 0=binary and 1=ascii file, but in RESOURCEs I saw always 0 */\
\
P2 = ReadByte();\
	/* must be 4,5 or 6 and it is a parameter for the decompression algorithm */\
\
\
/* Now, a bit stream follows, which is decoded as described below: */\
\
\
LOOP:\
     read 1 bit (take bits from the lowest value (LSB) to the MSB i.e. bit 0, bit 1 etc ...)\
         - if the bit is 0 read 8 bits and write it to the output as it is.\
         - if the bit is 1 we have here a length/distance pair:\
                 - decode a number with Hufmman Tree #1; variable bit length, result is 0x00 .. 0x0F -&#62; L1\
                   if L1 &#60;= 7:\
                         LENGTH = L1 + 2\
                   if L1 &#62; 7\
                         read more (L1-7) bits -&#62; L2\
                         LENGTH = L2 + M[L1-7] + 2\
\
                 - decode another number with Hufmann Tree #2 giving result 0x00..0x3F -&#62; D1\
                   if LENGTH == 2\
                         D1 = D1 &#60;&#60; 2\
                         read 2 bits -&#62; D2\
                   else\
                         D1 = D1 &#60;&#60; P2  // the parameter 2\
                         read P2 bits -&#62; D2\
\
                   DISTANCE = (D1 | D2) + 1\
\
                 - now copy LENGTH bytes from (output_ptr-DISTANCE) to output_ptr\
END LOOP\
\pard\pardeftab720\sa400\partightenfactor0

\f0 \cf0 \cb2 The algorithm terminates as soon as it runs out of bits. The data structures used are as follows:\cb1 \
\pard\pardeftab720\partightenfactor0

\f1\b\fs29\fsmilli14732 \cf0 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb2 M
\f0\b0\fs25\fsmilli12700 \cb1 \
\pard\pardeftab720\sa400\partightenfactor0
\cf0 \cb2 M is a constant array defined as M[0] = 7, M[n+1] = M[n]+ 2n. That means M[1] = 8, M[2] = 0x0A, M[3] = 0x0E, M[4] = 0x16, M[5] = 0x26, etc.\cb1 \
\pard\pardeftab720\partightenfactor0

\f1\b\fs29\fsmilli14732 \cf0 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb2 Huffman Tree #1
\f0\b0\fs25\fsmilli12700 \cb1 \
\pard\pardeftab720\sa400\partightenfactor0
\cf0 \cb2 The first huffman tree ({\field{\*\fldinst{HYPERLINK "https://wiki.sierrahelp.com/index.php/SCI_Specifications:_Chapter_2_-_Resource_files#Decompression_algorithm_HUFFMAN"}}{\fldrslt \cf8 \strokec8 the Section called\'a0
\f3\i Decompression algorithm HUFFMAN}}) contains the length values. It is described by the following table:\cb1 \
\pard\pardeftab720\partightenfactor0
\cf0 \

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrt\brdrs\brdrw20\brdrcf6 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clcbpat5 \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clcbpat5 \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\qc\partightenfactor0

\f1\b \cf0 \cb2 value (hex)\cb1 \cell 
\pard\intbl\itap1\pardeftab720\qc\partightenfactor0
\cf0 \cb2 code (binary)\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0

\f0\b0 \cf0 \cb2 0\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 11\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 2\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 3\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 4\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 5\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 6\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 7\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0010 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 8\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0010 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 9\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 a\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 b\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 11\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 c\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 10\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 d\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 01\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 e\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrb\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 f\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 000\cb1 \cell \lastrow\row
\pard\pardeftab720\sa400\partightenfactor0
\cf0 \cb2 where bits should be read from the left to the right.\cb1 \
\pard\pardeftab720\partightenfactor0

\f1\b\fs29\fsmilli14732 \cf0 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb2 Huffman Tree #2
\f0\b0\fs25\fsmilli12700 \cb1 \
\pard\pardeftab720\sa400\partightenfactor0
\cf0 \cb2 The second huffman code tree contains the distance values. It can be built from the following table:\cb1 \
\pard\pardeftab720\partightenfactor0
\cf0 \

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrt\brdrs\brdrw20\brdrcf6 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clcbpat5 \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clcbpat5 \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\qc\partightenfactor0

\f1\b \cf0 \cb2 value (hex)\cb1 \cell 
\pard\intbl\itap1\pardeftab720\qc\partightenfactor0
\cf0 \cb2 code (binary)\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0

\f0\b0 \cf0 \cb2 00\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 11\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 01\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 02\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 03\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1001 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 04\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1001 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 05\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1000 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 06\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1000 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 07\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0111 11\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 08\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0111 10\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 09\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0111 01\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0a\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0111 00\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0b\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0110 11\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0c\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0110 10\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0d\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0110 01\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0e\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0110 00\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0f\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0101 11\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 10\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0101 10\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 11\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0101 01\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 12\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0101 00\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 13\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0100 11\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 14\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0100 10\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 15\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0100 01\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 16\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0100 001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 17\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0100 000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 18\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0011 111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 19\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0011 110\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1a\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0011 101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1b\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0011 100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1c\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0011 011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1d\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0011 010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1e\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0011 001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1f\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0011 000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 20\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0010 111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 21\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0010 110\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 22\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0010 101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 23\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0010 100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 24\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0010 011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 25\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0010 010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 26\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0010 001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 27\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0010 000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 28\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 29\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 110\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 2a\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 2b\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 2c\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 2d\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 2e\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 2f\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 30\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 31\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1110\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 32\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 33\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 34\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 35\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 36\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 37\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 38\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 39\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0110\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 3a\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 3b\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 3c\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 3d\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 3e\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrb\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth1595\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 3f\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000\cb1 \cell \lastrow\row
\pard\pardeftab720\sa400\partightenfactor0
\cf0 \cb2 where bits should be read from the left to the right.\cb1 \
\pard\pardeftab720\partightenfactor0

\f1\b\fs29\fsmilli14732 \cf0 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb2 Huffman Tree #3
\f0\b0\fs25\fsmilli12700 \cb1 \
\pard\pardeftab720\sa400\partightenfactor0
\cf0 \cb2 This tree describes literal values for ASCII mode, which adds another compression step to the algorithm.\cb1 \
\pard\pardeftab720\partightenfactor0
\cf0 \

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrt\brdrs\brdrw20\brdrcf6 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clcbpat5 \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clcbpat5 \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\qc\partightenfactor0

\f1\b \cf0 \cb2 value (hex)\cb1 \cell 
\pard\intbl\itap1\pardeftab720\qc\partightenfactor0
\cf0 \cb2 code (binary)\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0

\f0\b0 \cf0 \cb2 00\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1001 001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 01\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0111 1111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 02\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0111 1110\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 03\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0111 1101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 04\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0111 1100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 05\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0111 1011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 06\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0111 1010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 07\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0111 1001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 08\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0111 1000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 09\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 1101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0a\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0100 011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0b\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0111 0111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0c\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0111 0110\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0d\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0100 010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0e\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0111 0101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0f\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0111 0100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 10\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0111 0011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 11\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0111 0010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 12\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0111 0001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 13\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0111 0000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 14\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0110 1111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 15\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0110 1110\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 16\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0110 1101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 17\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0110 1100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 18\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0110 1011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 19\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0110 1010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1a\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0010 0100 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1b\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0110 1001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1c\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0110 1000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1d\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0110 0111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1e\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0110 0110\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1f\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0110 0101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 20\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 21\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1010 01\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 22\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 1100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 23\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0110 0100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 24\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1010 00\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 25\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0110 0011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 26\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1001 11\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 27\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 1011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 28\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0100 001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 29\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0100 000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 2a\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 1010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 2b\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1101 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 2c\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0011 111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 2d\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1001 01\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 2e\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0011 110\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 2f\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 1001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 30\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0011 101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 31\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1001 00\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 32\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0011 100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 33\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0011 011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 34\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0011 010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 35\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0011 001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 36\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 1000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 37\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0011 000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 38\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0010 111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 39\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 0111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 3a\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 0110\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 3b\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0110 0010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 3c\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1001 000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 3d\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0010 110\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 3e\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1101 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 3f\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1000 111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 40\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0110 0001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 41\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1000 11\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 42\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0010 101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 43\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1000 10\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 44\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1000 01\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 45\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1110 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 46\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0010 100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 47\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 0101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 48\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 0100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 49\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1000 00\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 4a\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1000 110\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 4b\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1100 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 4c\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0111 11\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 4d\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0010 011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 4e\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0111 10\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 4f\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0111 01\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 50\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0010 010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 51\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1000 101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 52\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0111 00\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 53\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0110 11\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 54\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0110 10\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 55\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0010 001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 56\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1100 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 57\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 0011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 58\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1011 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 59\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1011 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 5a\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1000 100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 5b\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 0010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 5c\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1000 011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 5d\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1010 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 5e\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0110 0000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 5f\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 0001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 60\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0101 1111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 61\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1110 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 62\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0110 01\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 63\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0110 00\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 64\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0101 11\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 65\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1101 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 66\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0101 10\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 67\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0101 01\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 68\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0101 00\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 69\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1101 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 6a\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1000 010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 6b\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0010 000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 6c\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1100 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 6d\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0100 11\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 6e\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1100 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 6f\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1011 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 70\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0100 10\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 71\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1001 10\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 72\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1011 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 73\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1010 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 74\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1010 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 75\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 1001 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 76\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 0000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 77\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0001 111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 78\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 79\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1110\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 7a\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1001 01\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 7b\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1000 001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 7c\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 1000 000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 7d\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0101 1110\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 7e\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0101 1101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 7f\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0101 1100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 80\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0010 0100 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 81\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0010 0011 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 82\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0010 0011 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 83\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0010 0010 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 84\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0010 0010 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 85\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0010 0001 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 86\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0010 0001 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 87\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0010 0000 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 88\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0010 0000 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 89\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 1111 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 8a\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 1111 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 8b\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 1110 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 8c\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 1110 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 8d\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 1101 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 8e\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 1101 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 8f\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 1100 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 90\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 1100 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 91\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 1011 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 92\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 1011 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 93\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 1010 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 94\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 1010 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 95\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 1001 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 96\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 1001 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 97\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 1000 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 98\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 1000 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 99\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 0111 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 9a\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 0111 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 9b\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 0110 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 9c\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 0110 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 9d\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 0101 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 9e\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 0101 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 9f\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 0100 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 a0\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 0100 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 a1\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 0011 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 a2\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 0011 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 a3\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 0010 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 a4\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 0010 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 a5\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 0001 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 a6\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 0001 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 a7\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 0000 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 a8\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0001 0000 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 a9\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 1111 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 aa\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 1111 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 ab\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 1110 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 ac\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 1110 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 ad\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 1101 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 ae\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 1101 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 af\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 1100 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 b0\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0101 1011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 b1\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0101 1010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 b2\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0101 1001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 b3\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0101 1000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 b4\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0101 0111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 b5\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0101 0110\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 b6\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0101 0101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 b7\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0101 0100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 b8\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0101 0011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 b9\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0101 0010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 ba\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0101 0001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 bb\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0101 0000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 bc\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0100 1111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 bd\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0100 1110\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 be\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0100 1101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 bf\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0100 1100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 c0\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0100 1011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 c1\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0100 1010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 c2\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0100 1001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 c3\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0100 1000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 c4\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0100 0111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 c5\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0100 0110\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 c6\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0100 0101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 c7\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0100 0100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 c8\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0100 0011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 c9\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0100 0010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 ca\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0100 0001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 cb\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0100 0000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 cc\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0011 1111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 cd\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0011 1110\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 ce\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0011 1101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 cf\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0011 1100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 d0\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0011 1011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 d1\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0011 1010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 d2\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0011 1001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 d3\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0011 1000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 d4\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0011 0111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 d5\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0011 0110\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 d6\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0011 0101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 d7\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0011 0100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 d8\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0011 0011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 d9\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0011 0010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 da\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0011 0001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 db\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0011 0000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 dc\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0010 1111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 dd\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0010 1110\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 de\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0010 1101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 df\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0010 1100\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 e0\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 1100 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 e1\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0010 1011\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 e2\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 1011 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 e3\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 1011 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 e4\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 1010 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 e5\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0010 1010\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 e6\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 1010 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 e7\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 1001 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 e8\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 1001 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 e9\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0010 1001\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 ea\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 1000 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 eb\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 1000 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 ec\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 0111 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 ed\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 0111 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 ee\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0010 1000\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 ef\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 0110 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 f0\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 0110 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 f1\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 0101 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 f2\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0010 0111\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 f3\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0010 0110\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 f4\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0010 0101\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 f5\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 0101 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 f6\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 0100 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 f7\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 0100 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 f8\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 0011 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 f9\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 0011 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 fa\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 0010 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 fb\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 0010 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 fc\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 0001 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 fd\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 0001 0\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 fe\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 0000 1\cb1 \cell \row

\itap1\trowd \taflags1 \trgaph108\trleft-108 \trcbpat4 \tamart253 \tamarb253 \trbrdrl\brdrs\brdrw20\brdrcf6 \trbrdrb\brdrs\brdrw20\brdrcf6 \trbrdrr\brdrs\brdrw20\brdrcf6 
\clvertalc \clshdrawnil \clwWidth1327\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx4320
\clvertalc \clshdrawnil \clwWidth2048\clftsWidth3 \clbrdrt\brdrs\brdrw20\brdrcf6 \clbrdrl\brdrs\brdrw20\brdrcf6 \clbrdrb\brdrs\brdrw20\brdrcf6 \clbrdrr\brdrs\brdrw20\brdrcf6 \clpadt50 \clpadl101 \clpadb50 \clpadr101 \gaph\cellx8640
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 ff\cb1 \cell 
\pard\intbl\itap1\pardeftab720\partightenfactor0
\cf0 \cb2 0000 0000 0000 0\cb1 \cell \lastrow\row
\pard\pardeftab720\sa400\partightenfactor0
\cf0 \cb2 where bits should be read from the left to the right.\cb1 \
\pard\pardeftab720\partightenfactor0

\f1\b\fs32\fsmilli16256 \cf0 \cb2 Decompression algorithm UNKNOWN
\f0\b0\fs25\fsmilli12700 \cb1 \
\pard\pardeftab720\sa400\partightenfactor0
\cf0 \cb2 The algorithms listed as UNKNOWN-x have not yet been mapped to actual algorithms but are known to be used by the games. For some of them, it is possible that they match one of the algorithms described above, but have not yet been added to FreeSCI in an appropriate way (refer to DCL-EXPLODE for a good example).\cb1 \
}