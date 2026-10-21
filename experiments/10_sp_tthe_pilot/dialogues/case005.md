# Situation Puzzles case 5 — bare vs evolved
**Why this case:** 进化版提问质量提升最大
Same puzzle, same frozen Qwen2.5-7B questioner, same official Llama-3.1-405B referee, same 25-question budget. The only difference is the harness around the model. Records come from the frozen evaluation of all 100 test puzzles (`meta/cd_tthe/logs/eval_sp100/`), not from the evolution stream.
### bare — the official AR-Bench zero-shot loop

F1 0.541 · key-question recall 0.00 (0/6) · 690 output tokens · 25 questions

 1. `Q: Was Emily's family living in a remote area with limited postal services?` → **Unknown**
 2. `Q: Did Emily leave any electronic devices behind that could be used to send a message?` → **No**
 3. `Q: Had Emily's family recently adopted a new member who is skilled in impersonation or acting?` → **No**
 4. `Q: Was the letter handwritten by someone who knew Emily well?` → **No**
 5. `Q: Did Emily's family receive any unusual packages or letters before the funeral that might have contained instructions or clues?` → **No**
 6. `Q: Was Emily involved in any theatrical productions or had experience in performance arts?` → **Unknown**
 7. `Q: Could it be possible that Emily's family found the letter in a place unrelated to the funeral, such as their home or workplace?` → **No**
 8. `Q: Did Emily's family discover that the letter was part of a prank or a practical joke played on them?` → **No**
 9. `Q: Was the letter sent via an alternative method, such as a courier service that was not immediately noticed as coming from the funeral home?` → **No**
10. `Q: Did Emily leave any instructions or hints about how to contact her after her passing?` → **No**
11. `Q: Was the letter delivered to the family through a means other than traditional mail, such as a friend or acquaintance?` → **No**
12. `Q: Did Emily's family receive the letter at a location different from where they were when they thought they had completed all funeral arrangements?` → **Yes**
13. `Q: Was the letter delivered to a different address or location than where the funeral took place?` → **Unknown**
14. `Q: Did Emily's family receive the letter at a location where they were not expecting it, such as a second home or a relative's house?` → **No**
15. `Q: Was the letter delivered to a location that was not directly related to the funeral preparations but coincidentally close to where the family was during the funeral?` → **No**
16. `Q: Did Emily's family receive the letter at a location where they were attending a social event or gathering unrelated to the funeral?` → **No**
17. `Q: Was the letter delivered to a location that was mentioned or discussed during the funeral, leading the family to check there later?` → **No**
18. `Q: Did Emily's family receive the letter at a location where they had previously stored important documents or items related to the funeral?` → **Unknown**
19. `Q: Was the letter delivered to a location where one of the family members had gone to run an errand or perform a task unrelated to the funeral?` → **No**
20. `Q: Did Emily's family receive the letter at a location that was not the primary focus of their attention during the funeral proceedings?` → **Yes**
21. `Q: Was the letter delivered to a secondary location, such as a relative's home or a friend's house, where one of the family members was staying temporarily?` → **No**
22. `Q: Did Emily's family receive the letter at a location that was mentioned in the eulogy or during the funeral service?` → **Yes**
23. `Q: Was the location where the family received the letter during the funeral mentioned as a point of reference or a place to gather after the service?` → **No**
24. `Q: Did the person delivering the letter mention anything during the funeral that might have led the family to check another location for the letter?` → **No**
25. `Q: Was the letter left in a conspicuous place at the funeral venue that the family overlooked during their initial search?` → **No**

**Final answer**

> A: The letter was delivered to a location that was mentioned in the eulogy or during the funeral service, but the family did not immediately think to check there. After the service, they likely revisited the location or returned to the funeral venue, where they found the letter.

### b0r1g2 — first committed harness

F1 0.860 · key-question recall 0.17 (1/6) · 3471 output tokens · 25 questions

 1. `Q: Was a family member involved in writing the letter?` → **No**
 2. `Q: Did Emily die during her funeral?` → **No**
 3. `Q: Did the family attend her burial?` → **Yes**
 4. `Q: Was the letter written before Emily's death?` → **No**
 5. `Q: Did the letter come from someone who was at the funeral but is still alive?` → **No**
 6. `Q: Was the hologram activated by a family member?` → **No**
 7. `Q: Was Emily involved in arranging to send the letter?` → **Yes**
 8. `Q: Did Emily trigger the playback of the video herself?` → **No**
 9. `Q: Was the video stored on a device that was with Emily and accessed after her death?` → **No**
10. `Q: Did Emily use a pre-programmed device that activated automatically?` → **No**
11. `Q: Was the letter sent using a pre-programmed mechanism?` → **No**
12. `Q: Was a friend involved in programming the smartwatch for Emily?` → **No**
13. `Q: Did the family notice anything unusual during the burial that might indicate Emily had made arrangements beforehand?` → **No**
14. `Q: Was the letter sent on purpose?` → **Yes**
15. `Q: Did Emily trigger the message remotely?` → **No**
16. `Q: Did Emily want the message to be sent exactly two days after her death?` → **Unknown**
17. `Q: Was the smartwatch programmed by Emily before her death?` → **No**
18. `Q: Was the letter sent by a stranger?` → **No**
19. `Q: Did a family member program the smartwatch?` → **Unknown**
20. `Q: Did Emily set a specific date and time trigger for the letter?` → **No**
21. `Q: Was the letter meant as a surprise?` → **Yes**
22. `Q: Was a trusted family member involved in activating the smartwatch to send the letter?` → **No**
23. `Q: Was the smartwatch programmed with a delay trigger?` → **No**
24. `Q: Did the family find any signs that Emily might have kept her arrangement with the smartwatch secret during the burial?` → **Unknown**
25. `Q: Was the letter sent using a delayed activation mechanism?` → **No**

**Final answer**

> During the burial, Emily's family did not notice anything unusual because she had secretly programmed a smartwatch to send the letter two days after her death. Emily arranged this as a surprise. She wanted to ensure that the letter would be sent exactly two days after her funeral, so she set a specific date and time trigger for the smartwatch. Emily was involved in arranging the timing, and the letter was meant as a surprise to her family. The family attended her burial, and the smartwatch was programmed by Emily before her death, ensuring the letter was sent on purpose.

### b2r1g0 — final committed harness

F1 0.800 · key-question recall 1.00 (6/6) · 4202 output tokens · 25 questions

 1. `Q: Was a family member involved in writing the letter?` → **No**
 2. `Q: Did Emily die during her funeral?` → **No**
 3. `Q: Did the family attend her burial?` → **Yes**
 4. `Q: Was the letter written before Emily's death?` → **No**
 5. `Q: Did the letter come from someone who was at the funeral but is still alive?` → **No**
 6. `Q: Was a twin involved in Emily's funeral?` → **No**
 7. `Q: Was Emily involved in arranging the letter?` → **Yes**
 8. `Q: Was there any video message involved in the letter?` → **No**
 9. `Q: Did Emily choose to use a delayed message for a specific reason?` → **No**
10. `Q: Was there any timed release mechanism involved in sending the letter?` → **No**
11. `Q: Did Emily use a trick to make it seem like she wrote a post-funeral letter?` → **No**
12. `Q: Was there any GPS-triggered mailbox involved in Emily's letter?` → **No**
13. `Q: Emily recorded the letter herself?` → **Yes**
14. `Q: Did Emily use a special device to send the letter two days after her recorded message?` → **No**
15. `Q: Did Emily use a specific trigger to delay the letter's playback?` → **No**
16. `Q: Was the letter recorded in Emily's home studio?` → **No**
17. `Q: Was the letter intentionally delayed?` → **No**
18. `Q: Did Emily use a simple timer to delay the playback of the recording?` → **No**
19. `Q: Did Emily know her family would attend her funeral and set up the recorder accordingly?` → **No**
20. `Q: Was a stranger involved in sending the letter?` → **No**
21. `Q: Did Emily use a delayed playback mechanism to record the letter?` → **No**
22. `Q: Did Emily set a specific trigger to delay the letter's delivery?` → **No**
23. `Q: Was the letter meant to comfort her family?` → **Yes**
24. `Q: Emily set up an audio device with a delayed timer?` → **No**
25. `Q: Was the letter intended to provide comfort to her family after her passing?` → **No**

**Final answer**

> Emily arranged for a pre-recorded letter to be sent to her family via mail, which was delivered two days after her funeral. She knew her family would attend her funeral and set up the recorder accordingly. The letter was meant to comfort her family. No one else was involved in writing or sending the letter, and it was not a delayed message triggered by any external factors. Emily recorded the letter herself and ensured it was mailed out just before her funeral, so it would arrive two days later.
