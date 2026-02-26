# Third & 20 - Output Validation Guide

**Purpose:** How to look at pipeline output and know if it's right.
**Audience:** You. A scientist who watches football. Not a coaching manual.

---

## The Rule

Your system measures **geometry**. It does not diagnose **scheme**.

When you validate, you're checking: *"Did the system count correctly and put players in the right zones?"*

You are NOT checking: *"Is this Cover 3?"* That's interpretation. That comes later. That's the coach's job.

---

## What to Check Per Play

### 1. Player Count

**What to look for:** `total_defensive_players` in the output CSV.

| Value | Verdict |
|-------|---------|
| 10-11 | Good. Full defense detected. |
| 8-9 | Acceptable. Lost 2-3 to occlusion at LOS (normal for HS film). |
| 6-7 | Suspicious. Check if camera angle cuts off part of field. |
| <6 or >14 | Broken. Detection or tracking failed on this clip. |

**How to verify:** Pause the video at snap frame. Count bodies on the defensive side of the ball. Compare to the number in the CSV.

---

### 2. Box Count

**What the system outputs:** Number of defenders in the "tackle box" area near the LOS.

**What to look for:** `box_count` in defensive_analysis CSV.

| Value | What it means | When it's normal |
|-------|--------------|-----------------|
| 5-6 | Light box | Passing situations, 2-high safety looks |
| 7 | Standard | Most common at HS level |
| 8+ | Loaded/stacked | Short yardage, goal line, pressure looks |
| <5 or >9 | Suspicious | Check if LOS is placed correctly |

**How to verify:** Look at the snap frame. Count defenders who are:
- Between the offensive tackles (width-wise)
- Within ~5 yards of the line of scrimmage (depth-wise)

That count should roughly match `box_count`. Off by 1 is normal (boundary player judgment calls). Off by 3+ means LOS or depth thresholds are wrong.

---

### 3. Defenders on Line of Scrimmage

**What to look for:** `defenders_on_los` in the CSV.

| Value | What it usually means |
|-------|----------------------|
| 3 | 3-down front (3-4, 3-3-5) |
| 4 | 4-down front (4-3, 4-2-5) |
| 5+ | Goal line / heavy package |
| <3 | Unusual or detection error |

**How to verify:** Count the defenders who are lined up directly across from offensive linemen, in a stance, at the line of scrimmage. Not standing up 3 yards back - those are linebackers.

This is the easiest thing to verify visually. If the system says 4 on the line and you see 4 big guys in three-point stances across from the O-line, it's working.

---

### 4. Deep Defenders (Safety Structure)

**What to look for:** `deep_defenders` and `safety_alignment` in the CSV.

| deep_defenders | safety_alignment | What you should see on film |
|---------------|-----------------|---------------------------|
| 0 | 0-high | Nobody deep. All defenders near the LOS. Rare. |
| 1 | 1-high | One safety standing alone, deep middle, 10-15+ yards off ball |
| 2 | 2-high | Two safeties deep, one on each side, roughly even depth |
| 1 + 1 shallow | robber | One safety deep, one close to LOS (8-ish yards) |

**How to verify:** Before the snap, look at the deepest defenders. How many are standing 10+ yards behind the line of scrimmage? That should match `deep_defenders`.

The distinction between 1-high and 2-high is the most visually obvious thing on the field. If you can see one guy standing alone deep in the middle → 1-high. Two guys splitting the deep field → 2-high. If the system gets this wrong consistently, something is off with depth calculation.

---

### 5. LOS Placement

**What to look for:** `los_x` in PlayState output (normalized 0-1 or in yards if calibrated).

**How to verify:** The LOS should be at the point where the two groups of players face each other. On endzone film, this is where the offensive and defensive lines meet.

If LOS is placed correctly, all downstream measurements (depth, box, safety classification) have a chance of being right. If LOS is wrong, everything downstream is wrong.

**Red flags:**
- LOS shifts dramatically between consecutive plays in the same drive
- LOS is placed at the edge of the frame (camera issue)
- Box count is 0 or 11+ (usually means LOS is in the wrong place)

---

### 6. Front Classification

**What to look for:** `front` in the CSV.

| Output | What to see |
|--------|-------------|
| 4-down | 4 guys in stances on the line. Most common at HS. |
| 3-down | 3 guys in stances. Often one big nose tackle over center. |
| 5-down | 5 on the line. Goal line / short yardage. |

**How to verify:** This is just `defenders_on_los` mapped to a label. If `defenders_on_los` is correct, this is correct.

---

## Validation Workflow

### Quick Scan (5 minutes per game)

1. Open the `_defensive_analysis.csv` for a game
2. Check the `total_defensive_players` column - should be 8-11 most plays
3. Check `box_count` - should vary between 5-8 across the game (if it's 7 every single play, something is stuck)
4. Check `safety_alignment` - should be a mix of 1-high and 2-high (if it's all one thing, check depth thresholds)
5. Check `defenders_on_los` - should be 3-5 most plays

### Visual Spot-Check (15 minutes per game)

1. Pick 5 plays at random
2. Open the video, go to the snap frame listed in the CSV
3. For each play, check:
   - Player count (rough match?)
   - Box count (±1 of what you see?)
   - Deep safety count (matches what you see?)
   - Front (matches linemen you count?)
4. Score: 4/5 or better = pipeline is working. 2/5 or worse = something upstream is broken.

### Detailed Audit (1 hour per game)

1. Pick 10-15 plays across different game situations (1st down, 3rd and long, red zone, etc.)
2. For each play, record:
   - What the system says (from CSV)
   - What you see (from video)
   - Whether they agree
3. Categorize disagreements:
   - **Count errors** (system says 7 in box, you count 5) → LOS or detection problem
   - **Boundary errors** (system says 7, you'd say 6 or 8) → threshold calibration, not broken
   - **Classification errors** (system says 2-high, you see 1-high) → depth threshold problem

---

## What "Good Enough for Demo" Looks Like

You don't need perfection. You need a coach to look at the output and say "yeah, that's mostly right."

| Metric | Target Accuracy |
|--------|----------------|
| Player count | Within 2 of actual, 80%+ of plays |
| Box count | Within 1 of actual, 70%+ of plays |
| Safety alignment (1-high vs 2-high) | Correct 70%+ of plays |
| Front (3-down vs 4-down) | Correct 75%+ of plays |

If you hit those numbers, you have a product. No existing tool automates these measurements from HS film without human tagging.

---

## Common Failure Modes

| Symptom | Likely Cause |
|---------|-------------|
| Everything is classified the same | LOS stuck at fixed position, not adapting to plays |
| Box count always 0 or always 11 | LOS is way off, all players on one side |
| Safety always 0-high | Deep defenders not detected (tracking lost them) |
| Front always 3-down | LOS placed too deep, linemen classified as second level |
| Player count consistently low | YOLO confidence threshold too high, or camera angle cutting off field |
| Wild variation in counts between similar plays | Tracking ID fragmentation (ByteTrack losing tracks) |

---

## What NOT to Validate

Do NOT try to validate:
- **Coverage scheme labels** (Cover 3, Cover 1, etc.) - these are coaching interpretations, not geometric facts
- **Play call predictions** - that's the recommendation engine, not the CV pipeline
- **Individual player identity** - jersey OCR is noisy, validate by position/color not number
- **Post-snap behavior** - current system is pre-snap geometry only for defense

---

## The Scientist's Checklist

Before showing output to anyone:

- [ ] Ran pipeline with PlayState integrated (not old circular LOS)
- [ ] Checked player count distribution (should center around 9-11)
- [ ] Verified box count varies game-to-game and within games
- [ ] Confirmed safety alignment isn't all one category
- [ ] Spot-checked 5 plays visually against video
- [ ] Confirmed LOS placement looks reasonable on spot-checked plays
- [ ] Documented any systematic errors found

If all boxes checked → you're ready for a coach to validate.
If systematic errors found → fix upstream (likely LOS or detection) before showing anyone.
