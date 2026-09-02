# College RAG Chatbot - Test Results

**Summary:** 19/19 test cases passed (100.0%)

| # | Category | Query | Expected | Actual | Pass | Latency (ms) |
|---|----------|-------|----------|--------|------|---------------|
| 1 | Normal - Admissions | What is the last date to apply for admission? | ok | ok | ✅ | 1.35 |
| 2 | Normal - Fees | How much is the B.Tech tuition fee per year? | ok | ok | ✅ | 0.92 |
| 3 | Normal - Courses | What is the intake for Computer Science and Engineering? | ok | ok | ✅ | 0.77 |
| 4 | Normal - Exams | What percentage of attendance is required for the semeste... | ok | ok | ✅ | 0.73 |
| 5 | Normal - Hostel | What time do I need to be back in the hostel on weekdays? | ok | ok | ✅ | 0.76 |
| 6 | Normal - Placements | What was the highest placement package last year? | ok | ok | ✅ | 0.76 |
| 7 | Paraphrased | When do I need to submit original documents after seat al... | ok | ok | ✅ | 0.58 |
| 8 | Paraphrased | Is there a fee waiver for good sports players? | ok | ok | ✅ | 0.65 |
| 9 | Paraphrased | Can I get my money back if I cancel my seat? | ok | ok | ✅ | 0.63 |
| 10 | Ambiguous / multi-topic | Tell me about fees and hostel rules | ok | ok | ✅ | 0.58 |
| 11 | Out-of-domain | What is the capital of France? | no_match | no_match | ✅ | 0.5 |
| 12 | Out-of-domain | Write a python program to sort a list | no_match | no_match | ✅ | 0.51 |
| 13 | Out-of-domain | Who won the cricket world cup in 2023? | no_match | no_match | ✅ | 0.46 |
| 14 | Edge - empty string | '' | empty_input | empty_input | ✅ | 0.0 |
| 15 | Edge - whitespace only |       | empty_input | empty_input | ✅ | 0.0 |
| 16 | Edge - gibberish/punctuation | !!!@@@###??? | invalid_input | invalid_input | ✅ | 0.0 |
| 17 | Edge - single word | fees | ok | ok | ✅ | 0.61 |
| 18 | Edge - very long input | fees fees fees fees fees fees fees fees fees fees fees fe... | too_long | too_long | ✅ | 0.0 |
| 19 | Edge - None (simulated bad client) | None | empty_input | empty_input | ✅ | 0.0 |

## Full answers

### 1. Normal - Admissions — `What is the last date to apply for admission?`
- **Status:** ok (expected: ok, PASS)
- **Answer:** A late payment fine of INR 100 per day applies after the due date, up to a maximum of 15 days, beyond which the student's registration may be cancelled. Candidates must apply through the official online admission portal by submitting scanned copies of their mark sheets, transfer certificate, migration certificate, and category certificate (if applicable). The non-refundable application fee is INR 1,000 for general category and INR 500 for SC/ST/PWD candidates. Tuition fees may be paid in two installments: the first at the time of admission and the second before the start of the even semester.
- **Sources:** admissions.txt (Application Process), admissions.txt (Important Dates for 2026-27), exams_and_library.txt (Revaluation and Retotaling), fees_and_scholarships.txt (Payment Schedule)

### 2. Normal - Fees — `How much is the B.Tech tuition fee per year?`
- **Status:** ok (expected: ok, PASS)
- **Answer:** Merit Scholarship: Students scoring above 90% in the qualifying examination receive a 25% tuition fee waiver for the first year, renewable each year based on maintaining a CGPA above 8.5. If a student withdraws admission before the commencement of classes, 90% of the tuition fee is refunded. Need-Based Scholarship: Available to students with family income below INR 3,00,000 per annum, covering up to 50% of tuition fees, subject to submission of income certificates. Sports Scholarship: State or national-level sports achievers can receive up to 100% tuition waiver, decided by the Sports Committee.
- **Sources:** courses_and_departments.txt (Postgraduate Programs), fees_and_scholarships.txt (Fees And Scholarships), fees_and_scholarships.txt (Refund Policy), fees_and_scholarships.txt (Scholarships Available)

### 3. Normal - Courses — `What is the intake for Computer Science and Engineering?`
- **Status:** ok (expected: ok, PASS)
- **Answer:** Computer Science and Engineering (CSE) - Intake: 120. Overall placement rate stood at 87% across all branches, with the Computer Science and Engineering department recording 96% placement. - M.Tech in Computer Science and Engineering (2 years). Electronics and Communication Engineering (ECE) - Intake: 90.
- **Sources:** courses_and_departments.txt (COURSES AND DEPARTMENTS - GREENFIELD COLLEGE OF ENGINEERING), courses_and_departments.txt (Postgraduate Programs), hostel_and_campus_life.txt (Hostel And Campus Life), placements.txt (Placement Statistics (2025-26))

### 4. Normal - Exams — `What percentage of attendance is required for the semester exam?`
- **Status:** ok (expected: ok, PASS)
- **Answer:** A minimum of 75% attendance is mandatory to be eligible to sit for the SEE; students with 65-74% attendance may apply for condonation with a valid medical certificate and a fine of INR 500. A minimum CGPA of 5.0 is required to be eligible for the degree without backlog clearance requirements. Each semester has two Continuous Internal Assessment (CIA) tests worth 15 marks each, an assignment/quiz component worth 10 marks, and a Semester End Examination (SEE) worth 60 marks, for a total of 100 marks per subject. Students with more than 3 active backlogs at the end of a semester are placed under academic probation and require mentor sign-off to continue.
- **Sources:** courses_and_departments.txt (Grading System), exams_and_library.txt (Examination Pattern)

### 5. Normal - Hostel — `What time do I need to be back in the hostel on weekdays?`
- **Status:** ok (expected: ok, PASS)
- **Answer:** - In-time for entry into the hostel is 8:00 PM on weekdays and 9:30 PM on weekends; late entries require prior written permission from the warden. The hostel mess provides breakfast, lunch, snacks, and dinner on a rotating weekly menu, with separate vegetarian and non-vegetarian counters. - Ragging in any form is strictly prohibited and punishable under the UGC anti-ragging regulations, including possible expulsion. - Visitors are permitted only in the designated visitor's lounge during visiting hours (4:00 PM - 6:00 PM).
- **Sources:** hostel_and_campus_life.txt (Hostel And Campus Life), hostel_and_campus_life.txt (Hostel Rules), hostel_and_campus_life.txt (Mess and Food), hostel_and_campus_life.txt (Sports Facilities)

### 6. Normal - Placements — `What was the highest placement package last year?`
- **Status:** ok (expected: ok, PASS)
- **Answer:** The highest package offered was INR 42 lakhs per annum (international offer), while the average package across the college was INR 6.8 lakhs per annum. Overall placement rate stood at 87% across all branches, with the Computer Science and Engineering department recording 96% placement. Over 150 companies visited the campus during the placement season. The Placement Cell also facilitates summer and semester-long internships, many of which convert into pre-placement offers (PPOs).
- **Sources:** placements.txt (Internship Support), placements.txt (Placement Statistics (2025-26)), placements.txt (Training and Preparation)

### 7. Paraphrased — `When do I need to submit original documents after seat allotment?`
- **Status:** ok (expected: ok, PASS)
- **Answer:** Candidates must report to the admission cell with original documents within 3 working days of seat allotment to confirm their seat. Failure to report within the stipulated time will result in forfeiture of the allotted seat, which will then be offered to the next candidate in the merit list. Counseling is conducted in three rounds.
- **Sources:** admissions.txt (Counseling and Seat Allotment)

### 8. Paraphrased — `Is there a fee waiver for good sports players?`
- **Status:** ok (expected: ok, PASS)
- **Answer:** Merit Scholarship: Students scoring above 90% in the qualifying examination receive a 25% tuition fee waiver for the first year, renewable each year based on maintaining a CGPA above 8.5. Sports Scholarship: State or national-level sports achievers can receive up to 100% tuition waiver, decided by the Sports Committee. Government Scholarships: The college facilitates applications for state and central government FEE STRUCTURE AND SCHOLARSHIPS - GREENFIELD COLLEGE OF ENGINEERING.
- **Sources:** fees_and_scholarships.txt (Fees And Scholarships), fees_and_scholarships.txt (Scholarships Available)

### 9. Paraphrased — `Can I get my money back if I cancel my seat?`
- **Status:** ok (expected: ok, PASS)
- **Answer:** Candidates must report to the admission cell with original documents within 3 working days of seat allotment to confirm their seat. Failure to report within the stipulated time will result in forfeiture of the allotted seat, which will then be offered to the next candidate in the merit list. Counseling is conducted in three rounds.
- **Sources:** admissions.txt (Counseling and Seat Allotment)

### 10. Ambiguous / multi-topic — `Tell me about fees and hostel rules`
- **Status:** ok (expected: ok, PASS)
- **Answer:** Hostel fees (including mess charges) are billed separately at INR 85,000 per annum for double-sharing rooms and INR 1,05,000 for single rooms. EXAMINATION RULES AND LIBRARY POLICY - GREENFIELD COLLEGE OF ENGINEERING. HOSTEL AND CAMPUS LIFE - GREENFIELD COLLEGE OF ENGINEERING. The hostel mess provides breakfast, lunch, snacks, and dinner on a rotating weekly menu, with separate vegetarian and non-vegetarian counters.
- **Sources:** exams_and_library.txt (Exams And Library), fees_and_scholarships.txt (Tuition Fees (Per Annum, 2026-27)), hostel_and_campus_life.txt (Hostel And Campus Life), hostel_and_campus_life.txt (Mess and Food)

### 11. Out-of-domain — `What is the capital of France?`
- **Status:** no_match (expected: no_match, PASS)
- **Answer:** I couldn't find anything about that in the college documents I have access to (admissions, fees, courses, exams, hostel/campus life, and placements). Could you rephrase your question, or ask about one of those topics?

### 12. Out-of-domain — `Write a python program to sort a list`
- **Status:** no_match (expected: no_match, PASS)
- **Answer:** I couldn't find anything about that in the college documents I have access to (admissions, fees, courses, exams, hostel/campus life, and placements). Could you rephrase your question, or ask about one of those topics?

### 13. Out-of-domain — `Who won the cricket world cup in 2023?`
- **Status:** no_match (expected: no_match, PASS)
- **Answer:** I couldn't find anything about that in the college documents I have access to (admissions, fees, courses, exams, hostel/campus life, and placements). Could you rephrase your question, or ask about one of those topics?

### 14. Edge - empty string — `''`
- **Status:** empty_input (expected: empty_input, PASS)
- **Answer:** Please type a question -- for example, "What is the last date to apply for admission?"

### 15. Edge - whitespace only — `     `
- **Status:** empty_input (expected: empty_input, PASS)
- **Answer:** Please type a question -- for example, "What is the last date to apply for admission?"

### 16. Edge - gibberish/punctuation — `!!!@@@###???`
- **Status:** invalid_input (expected: invalid_input, PASS)
- **Answer:** I couldn't understand that input. Please ask a question in words, e.g. "How do I apply for a scholarship?"

### 17. Edge - single word — `fees`
- **Status:** ok (expected: ok, PASS)
- **Answer:** Tuition fees may be paid in two installments: the first at the time of admission and the second before the start of the even semester. Hostel fees (including mess charges) are billed separately at INR 85,000 per annum for double-sharing rooms and INR 1,05,000 for single rooms. Need-Based Scholarship: Available to students with family income below INR 3,00,000 per annum, covering up to 50% of tuition fees, subject to submission of income certificates. A late payment fine of INR 100 per day applies after the due date, up to a maximum of 15 days, beyond which the student's registration may be cancelled.
- **Sources:** fees_and_scholarships.txt (Payment Schedule), fees_and_scholarships.txt (Scholarships Available), fees_and_scholarships.txt (Tuition Fees (Per Annum, 2026-27))

### 18. Edge - very long input — `fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees fees `
- **Status:** too_long (expected: too_long, PASS)
- **Answer:** That question is quite long -- could you shorten it to a single, specific question?

### 19. Edge - None (simulated bad client) — `None`
- **Status:** empty_input (expected: empty_input, PASS)
- **Answer:** Please type a question -- for example, "What is the last date to apply for admission?"
