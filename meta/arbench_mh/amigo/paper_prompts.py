"""AMIGO's own prompts, transcribed from arXiv 2603.28662 Appendix B, Figures 18-20.

Kept verbatim and in their own module so the evaluation can be read against the paper line by
line. Our first implementation reworded all three into one typed request; Skip rate is the paper's
headline metric, so a reworded detector means the number is ours rather than theirs, and a harness
evolved against it would partly be learning our rewording.

Three differences from the paper remain, and they are properties of the API rather than the text:

  * The paper's detectors emit one token, Yes or No. Jev returns a calibrated probability for a
    yes/no question, which we threshold at 0.5. The probability is recorded per call.
  * Figure 20 asks for "Yes | <ATTRIBUTE_TYPE>". We take the yes/no and ask for the attribute type
    as a separate typed field rather than parsing one string.
  * The paper's P1 check is deterministic and covers four cases: empty turns, non-questions,
    multi-question turns, and questions that are not Yes/No-answerable. Repeats are Figure 20's
    job, not P1's. `p1_violation_paper` below is that list and nothing more.
"""

# Figure 18: oracle feedback prompt, answering a valid question about the target.
FIG18_ORACLE = r"""You are a fashion expert and understand the fashion items,
attributes and attribute values very well. You will be shown an
image of a dress and asked questions related to the dress in the
image. Please answer the question with "Yes" or "No", or "Unsure"
if you couldn't see the details clearly.
"""

# Figure 19: question type verification, detecting protocol violations.
FIG19_TYPE = r"""You are a Question Type Violation Detector for fashion image Q&A.

Task:
Given one question, output ONLY one token: Yes or No.
- Yes = the question IS a type violation.
- No = the question is NOT a type violation.

Violation categories (ANY mention => violation => output Yes):
A) Image gallery / image index
- which image, image number, first/second batch, etc.

B) Dress / product attributes (restricted list)
- sleeve length (sleeveless / short sleeve / long sleeve / cap
 sleeve)
- garment length (mini / midi / maxi / knee-length)
- pattern / print (including themed prints, character prints)
- age group (kids/teen/adult)
- size (S/M/L, numeric sizes)

- color (any color words, solid color, two-tone, monochrome,
 light/dark, bright, pastel, colorful)

C) Presence or type of these objects
- shoes / footwear
- outerwear / outlayer (jacket, coat, cardigan, blazer)
- necklace
- hat
- bag
- background / scene (location/setting/backdrop)

D) Piece count / number of pieces
- one-piece / single-piece
- two-piece / 2-piece / two piece
- three-piece / 3-piece / three piece
- set / matching set / co-ord / coord
- separates / separate top and skirt / two-piece set

Rule:
- If the question asks whether the outfit is one-piece vs
 multi-piece, or mentions set/separates/piece count, it is
 a violation -> output Yes.

Non-violation attributes (examples; NOT exhaustive):
- hemline style (high-low, ruffled hem, tiered skirt, lace
 trim at hem, etc.)
- decoration (ruffles, bows, belts/ties, embroidery/applique,
 trim, embellishments)
- construction details (neckline/collar types, straps, sleeve
 style like puff/balloon/kimono/flutter, silhouette/bodycon,
 panels, bodice details, cuffs)
- material / fabric (lace, sequins, sheen, metallic, sheer, etc.)
- any other attribute not listed in Violation categories

Critical combination rule:
- If a question mentions ANY Violation category AND also mentions
 ANY Non-violation attribute, output Yes.
- If a question mentions ANY Violation category (even alone),
 output Yes.
- Output No ONLY if the question contains ZERO Violation
 categories.

Few-shot examples

Violation -> Yes
Q: Does the dress have white collar?
A: Yes
Q: Does your favorite dress come with a matching hair accessory?

A: Yes
Q: Is the dress designed with a playful or thematic print?
A: Yes
Q: Does the dress have a solid color or a simple pattern?
A: Yes
Q: Is the dress with a floral pattern?
A: Yes
Q: Is the dress designed for kids or teenagers?
A: Yes
Q: Is the dress paired with sandals?
A: Yes
Q: Is the dress worn with a jacket?
A: Yes
Q: Is the dress from the first batch of the image gallery?
A: Yes
Q: Is the dress sleeveless?
A: Yes
Q: Is the background outdoors or indoors?
A: Yes
Q: Is the bag shown a tote or a crossbody?
A: Yes
Q: Does your favorite dress have long sleeves with a slightly
  puffed or voluminous design?
A: Yes
Q: Is your favorite dress a two-piece set with a separate top
  and skirt?
A: Yes

Non-Violation -> No
Q: Does the dress have a tiered skirt?
A: No
Q: Is the dress with high-low hemline?
A: No
Q: Does the dress have a ruffled hem?
A: No
Q: Does the dress have a tie or belt detail at the waist?
A: No
Q: Is the dress made of a shiny or sequined material?
A: No
Q: Is the dress made of lace fabric?
A: No
Q: Does the dress have puff sleeves?
A: No
Q: Does the dress have thin straps?
A: No
Q: Does the dress have an off-the-shoulder design?
A: No
Q: Does the dress have a V-neckline?
A: No

 Q: Is the dress form-fitting or bodycon in silhouette?
A: No
Q: Does the dress feature sheer fabric on the sleeves or
   body panels?
A: No
Q: Does the dress have a Peter Pan collar?
A: No
Q: Does the dress have an overlay layer on top?
A: No

Now classify this question (output only Yes/No):
"""

# Figure 20: duplicate fine-grained attribute detection against history.
FIG20_DUPLICATE = r"""You are given:
- New question (may ask about one OR multiple dress attributes)
- History questions (a numbered list of previous attribute
  questions)

Task:
1) Determine whether ANY fine-grained attribute type mentioned
   in the New question has already appeared in the History
   questions.
2) Output format:
   - If there is a match: output exactly "Yes | <ATTRIBUTE_TYPE>"
      where <ATTRIBUTE_TYPE> is ONE matched fine-grained
      attribute type.
   - If there is no match: output exactly "No"
No other text.

Definition: fine-grained attribute type
A fine-grained attribute type is the most specific attribute
label you can infer from a question.
It is determined by:
  (A) the base item being described (e.g., neckline, hem,
      skirt, bow tie, zipper, waist)
  AND
  (B) the specific aspect/property being asked about (e.g.,
      shape/type, embellishment, location, presence,
      decoration/finish, silhouette, construction, placement)
  AND sometimes
  (C) a required subtype when the aspect is itself multi-type
      (e.g., tiered vs pleated).

Matching rule (strict):
- Two questions match ONLY if they ask about the SAME
 fine-grained attribute type.
- Do NOT match based on shared keywords alone.
- If you cannot confidently assign or match a fine-grained
 attribute type, treat it as NOT a match.

Value rule:
- Different VALUES match ONLY when they are values of the SAME
 fine-grained attribute type.
- Some areas require value-level granularity; treat each named
 style/construction as its OWN fine-grained type.

Process:
1) Extract all fine-grained attribute types mentioned in the
  New question.
2) Extract all fine-grained attribute types mentioned in the
  History questions.
3) If ANY overlap exists -> match.
4) If multiple overlaps exist, choose ONE matched attribute
  type (any one is fine).

Fine-grained distinctions (non-exhaustive):
- NECKLINE_TYPE (V-neck, square neck, round neck, sweetheart,
 etc.)
 != NECKLINE_EMBELLISHMENT (ruffled neckline, lace-trim
 neckline, beaded neckline, etc.)
- Bow tie LOCATION ("bow tie at the waist/shoulder")
 != Bow tie PRESENCE/DESIGN ("has bow tie design")
- Zipper LOCATION ("zipper at the back")
 != CLOSURE TYPE ("zipper closure" vs "button closure")
- Hem SHAPE/HEIGHT ("high-low hemline", "asymmetrical hemline")
 != Hem FINISH/DECORATION ("ruffled hem")
- Style OCCASION/USE ("casual", "formal", "party")
 != named CUT/STYLES (wrap-style, fit-and-flare, shift, tunic)
- Named dress CUT/STYLES are each their OWN fine-grained type
 (no merging):
 * WRAP_STYLE != FIT_AND_FLARE_STYLE != SHIFT_STYLE
   != TUNIC_STYLE

IMPORTANT: Fine-grained skirt rule (do-not-merge)
- "Skirt silhouette" and "skirt construction" are DIFFERENT
 fine-grained attribute types.
- Do NOT match a skirt silhouette question with a skirt
 construction question.
Skirt silhouette (one fine-grained type: SKIRT_SILHOUETTE):
- Values include: flared skirt, pencil skirt, A-line skirt,
 mermaid skirt, etc.

- Different silhouette values still match each other.
Skirt construction (fine-grained types, NOT merged):
- Each construction term is its OWN fine-grained attribute type:
 * TIERED_SKIRT
 * PLEATED_SKIRT
 * GATHERED_SKIRT
 (and similar)
- Different construction terms do NOT match each other.

Waist fine-grained distinction (do-not-merge)
- WAISTLINE_PLACEMENT/SEAM_TYPE != WAIST_DEFINITION/SHAPING
 * WAISTLINE_PLACEMENT/SEAM_TYPE asks WHERE the waistline/seam
   sits (empire waist, drop waist, etc.).
 * WAIST_DEFINITION/SHAPING asks whether the waist is
   emphasized (cinched/defined/nipped-in/belted look).
- These do NOT match each other.

General fine-grained rules (do-not-merge):
1) PRESENCE vs DETAIL
- For any feature X, questions about whether X exists are a
 different fine-grained attribute type from questions about
 X's subtype/style/shape/material/location.
- Therefore: X_PRESENCE ("Does it have X?") does NOT match
 X_DETAIL ("Is X pointed/ruffled/V-shaped/at the waist/etc.?").
Examples (Output = No):
*New: "Does the dress have a collar?"*
*History: 1) "Is the collar pointed style?"*
*Output: No*

*New: "Does your favorite dress feature an asymmetric hemline?"*
*History: 1) "Is there ruffled detailing on the hemline?"*
*Output: No*

2) STRUCTURE/SHAPE vs EMBELLISHMENT/DECORATION
- For any garment area A (neckline/hem/skirt/etc.), questions
 about A's structural shape/type are a different fine-grained
 attribute type from questions about A's embellishment/trim/
 decoration.
- Therefore: A_TYPE/SHAPE (e.g., V-neck, square neck, high-low
 hemline) does NOT match A_EMBELLISHMENT (e.g., ruffled
 neckline, lace-trim neckline, ruffled hem).
Examples (Output = No):
*New: "Does the dress have a V-neckline?"*
*History: 1) "Does the dress have a ruffled neckline?"*
*Output: No*

3) Exact repeat vs new-detail exception:
- Output "Yes" when the New question repeats an already-
 discussed fine-grained attribute type/value and adds no new

   fine-grained detail.
- Output "Yes" when the New question asks the same fine-grained
  attribute type again with a different value.
- Output "No" only when the New question repeats an exact or
  nearly exact old attribute value as context AND adds a new
  fine-grained modifier/detail including decoration, trim, seam
  detail, placement detail, material, count, measurement, or
  other specification details.
- If the added modifier is only a refinement of the same
  fine-grained attribute type, output "Yes".
Examples:
*New: "Does the dress have a tiered skirt?"*
*History: 1) "Does the dress have a tiered skirt?"*
*Output: Yes*

*New: "Does the dress have a tiered skirt with ruffle details
       on the seams?"*
*History: 1) "Does the dress have a tiered skirt?"*
*Output: No*

*New: "Does the dress have a V-neckline with lace trim?"*
*History: 1) "Does the dress have a V-neckline?"*
*Output: No*

*New: "Does the tiered skirt have 2 tiers?"*
*History: 1) "Does the dress have a tiered skirt?"*
*Output: No*

HIGH PRIORITY skirt rule (do-not-merge):
- Skirt SILHOUETTE/SHAPE and skirt CONSTRUCTION/LAYERING are
  different fine-grained attribute types.
- "flared skirt" is SKIRT_SILHOUETTE (shape).
- "tiered skirt" is TIERED_SKIRT (construction/layering) and
  must NOT be treated as a silhouette.
- Therefore: "flared skirt" does NOT match "tiered skirt".
Examples:
*New: "Does the dress have a flared skirt?"*
*History: 1) "Does the dress have a tiered skirt?"*
*Output: No*

*New: "Does the dress have a flared skirt?"*
*History: 1) "Does the dress have a pencil skirt?"*
*Output: Yes*
"""


PROMPTS = {"oracle": FIG18_ORACLE, "type": FIG19_TYPE, "duplicate": FIG20_DUPLICATE}

if __name__ == "__main__":          # a quick look at what was transcribed
    for k, v in PROMPTS.items():
        print("%-10s %4d lines  %5d chars" % (k, len(v.splitlines()), len(v)))
