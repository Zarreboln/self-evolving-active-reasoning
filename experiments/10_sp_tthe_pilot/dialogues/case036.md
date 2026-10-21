# Situation Puzzles case 36 — bare vs evolved
**Why this case:** F1 最高但关键问题一个没问到 — Sun 批评的那一类
Same puzzle, same frozen Qwen2.5-7B questioner, same official Llama-3.1-405B referee, same 25-question budget. The only difference is the harness around the model. Records come from the frozen evaluation of all 100 test puzzles (`meta/cd_tthe/logs/eval_sp100/`), not from the evolution stream.
### bare — the official AR-Bench zero-shot loop

F1 0.552 · key-question recall 0.00 (0/6) · 676 output tokens · 25 questions

 1. `Q: Was Emma's obituary published accidentally?` → **No**
 2. `Q: Did someone intentionally arrange for Emma's obituary to be published?` → **No**
 3. `Q: Was Emma's obituary part of a prank or joke?` → **No**
 4. `Q: Did Emma receive any prior indication or warning about the obituary before attending the memorial service?` → **No**
 5. `Q: Was there any recent news or events that might have led people to mistakenly believe Emma had passed away?` → **No**
 6. `Q: Did Emma have any close friends or family members who might have had access to the newspaper and intentionally placed the obituary?` → **No**
 7. `Q: Was Emma's obituary similar to another person's obituary that was recently published in the newspaper?` → **Unknown**
 8. `Q: Have there been any recent errors or mistakes in the newspaper regarding other obituaries?` → **Unknown**
 9. `Q: Was Emma's obituary published in a different section of the newspaper than usual for obituaries?` → **Unknown**
10. `Q: Was Emma's obituary published on a special edition or supplement of the newspaper?` → **Unknown**
11. `Q: Was Emma's obituary published on a weekend or holiday when the newspaper staff might have been less attentive?` → **No**
12. `Q: Did Emma's obituary include any unusual or misleading information that could have caused confusion?` → **No**
13. `Q: Was Emma's obituary published in a smaller local newspaper rather than a larger national one?` → **Unknown**
14. `Q: Was Emma's obituary published alongside an announcement or notice that might have contributed to the confusion?` → **No**
15. `Q: Did Emma's obituary mention any recent activities or events that might explain the reason behind its publication?` → **No**
16. `Q: Was Emma's obituary published in a language different from what is typically used in the newspaper?` → **No**
17. `Q: Did Emma's obituary include a fake signature or name that might have caused confusion?` → **No**
18. `Q: Was Emma's obituary published in a format that is uncommon for typical obituaries (e.g., a full-page ad instead of a brief notice)?` → **Unknown**
19. `Q: Did Emma's obituary include a misleading date or time that might have caused confusion?` → **No**
20. `Q: Was Emma's obituary published in a color section of the newspaper, which might have drawn attention in a way that seemed unusual?` → **Unknown**
21. `Q: Did Emma's obituary include any unusual details or information that were not typical for a standard obituary?` → **Unknown**
22. `Q: Was Emma's obituary published during a time when the newspaper had a higher volume of submissions or errors?` → **No**
23. `Q: Did Emma's obituary include a contact number or email address that might have been a clue to its purpose?` → **Unknown**
24. `Q: Was Emma's obituary published with a headline or title that might have suggested it was a real obituary?` → **Yes**
25. `Q: Did Emma's obituary include a headline that clearly stated it was a fake or a tribute rather than a real obituary?` → **No**

**Final answer**

> A: Emma's obituary was published with a headline that suggested it was a real obituary, which is why she was initially startled to find it in the newspaper. However, since there is no clear indication that it was marked as fake or a tribute, it appears that the intention was to make it seem real, possibly for a specific purpose such as a prank, a test of the newspaper's systems, or another undisclosed reason.

### b0r1g2 — first committed harness

F1 0.606 · key-question recall 0.00 (0/6) · 4552 output tokens · 25 questions

 1. `Q: Did someone else write Emma's obituary?` → **Yes**
 2. `Q: Was Emma alive before the memorial service was held?` → **Yes**
 3. `Q: Was the obituary published without any mistakes by the newspaper staff?` → **Yes**
 4. `Q: Was Emma still alive when the obituary was published?` → **Yes**
 5. `Q: Was the obituary written due to a tragic accident?` → **Yes**
 6. `Q: Was the colleague who wrote the obituary a friend of Emma?` → **Unknown**
 7. `Q: Did the tragic accident involve a car?` → **Unknown**
 8. `Q: Was the tragic accident involving Emma fatal?` → **No**
 9. `Q: Did Emma set up the tragic accident?` → **No**
10. `Q: Was a colleague involved in writing the obituary?` → **Unknown**
11. `Q: Was the tragic accident witnessed by the colleague who wrote the obituary?` → **No**
12. `Q: Was the tragic accident that led to the obituary fake?` → **No**
13. `Q: Was the colleague who wrote the obituary a person known to Emma?` → **Unknown**
14. `Q: Did the newspaper staff publish the obituary without verifying Emma's status?` → **No**
15. `Q: Was the obituary published intentionally?` → **No**
16. `Q: Was the tragic accident that led to the obituary a result of a natural disaster?` → **No**
17. `Q: Was the tragic accident non-fatal but severe enough to prompt an obituary?` → **No**
18. `Q: Was the person who wrote the obituary a stranger?` → **Unknown**
19. `Q: Was the tragic accident that led to Emma's obituary a result of a fatal incident?` → **No**
20. `Q: Was the tragic accident that led to the obituary unexpected and sudden?` → **No**
21. `Q: Was the obituary written to protect someone?` → **No**
22. `Q: Was Emma's injury non-fatal but severe?` → **No**
23. `Q: Did the colleague have incorrect medical information about Emma's condition?` → **Unknown**
24. `Q: Was the obituary based on incorrect information?` → **No**
25. `Q: Did the colleague who wrote the obituary believe the accident was fatal?` → **Unknown**

**Final answer**

> A colleague wrote Emma's obituary after witnessing a tragic accident that severely injured Emma, leading to her hospitalization in critical condition. The colleague, believing the accident to be fatal, published the obituary without checking Emma's actual status. Emma was alive and well when the obituary was published, and she attended the memorial service held in her honor. The newspaper staff published the obituary without any mistakes, despite Emma's continued life.

### b2r1g0 — final committed harness

F1 0.764 · key-question recall 0.00 (0/6) · 6808 output tokens · 25 questions

 1. `Q: Did someone else write Emma's obituary?` → **Yes**
 2. `Q: Was Emma alive before the memorial service was held?` → **Yes**
 3. `Q: Was the obituary published without any mistakes by the newspaper staff?` → **Yes**
 4. `Q: Was Emma still alive when the obituary was published?` → **Yes**
 5. `Q: Was the obituary written due to a tragic accident?` → **Yes**
 6. `Q: Was the friend who wrote the obituary known to Emma?` → **Unknown**
 7. `Q: Was the tragic accident related to a car crash?` → **Unknown**
 8. `Q: Did others witness the accident?` → **Unknown**
 9. `Q: Did Emma arrange her own obituary?` → **No**
10. `Q: Was the close friend who wrote the obituary a family member of Emma?` → **Unknown**
11. `Q: Was the accident related to a fall?` → **Unknown**
12. `Q: Was the obituary planned by someone else?` → **No**
13. `Q: Was the accident a fatal incident?` → **Yes**
14. `Q: Was the tragic accident that led to the obituary a natural disaster?` → **No**
15. `Q: Was the tragic accident that led to the obituary caused by a person?` → **Yes**
16. `Q: Did the newspaper staff check if Emma was still alive before publishing the obituary?` → **Unknown**
17. `Q: Did the newspaper verify Emma's status before publishing the obituary?` → **No**
18. `Q: Did the person responsible for the tragic accident have a personal relationship with Emma?` → **Yes**
19. `Q: Was there a close friend of Emma's who witnessed the fatal accident?` → **No**
20. `Q: Was the memorial service organized by the same person who wrote the obituary?` → **Unknown**
21. `Q: Did the newspaper staff receive the obituary before Emma's memorial service was held?` → **Yes**
22. `Q: Was the obituary submitted to the newspaper directly by Emma's estranged brother?` → **Unknown**
23. `Q: Was the fatal accident caused by a sibling rivalry?` → **Unknown**
24. `Q: Was the fatal accident caused by the person with a personal relationship to Emma acting out of spite?` → **No**
25. `Q: Was the obituary submitted with intent to deceive?` → **No**

**Final answer**

> acquaintance, driven by guilt and a desire to protect Emma from harm, wrote and submitted the obituary to the newspaper before Emma's memorial service. The obituary was published without any mistakes by the newspaper staff, leading to Emma's shock and confusion upon attending the memorial service. The tragic accident was a fatal incident caused by the acquaintance acting out of a personal vendetta against another individual, but the acquaintance did not act out of spite towards Emma. The newspaper received the obituary before Emma's memorial service was held, unaware that Emma was still alive.
