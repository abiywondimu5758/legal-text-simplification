# Amharic Legal Sentence Simplification – Style Guide

## 1. Purpose

This style guide defines how complex Amharic legal sentences must be rewritten into plain Amharic while preserving full legal meaning.

The goal is **clarity without loss of legal force**.  
This guide applies to all 2,000 sentence pairs in the dataset and must be followed strictly.

---

## 2. Task Definition

Given a **single complex Amharic legal sentence**, produce a **simpler Amharic sentence or sentences** that:

- Preserve the exact legal meaning
- Are easier to read and understand
- Use common Amharic vocabulary
- Maintain legal obligations, permissions, and prohibitions

This is **not** summarization, explanation, or translation.

---

## 3. Core Principles (Non-Negotiable)

### 3.1 Meaning Preservation

The simplified sentence MUST:

- Preserve all legal obligations (አለበት, ይገባል)
- Preserve all prohibitions (አይፈቀድም)
- Preserve all permissions (ይፈቀዳል)
- Preserve all conditions (ከ… በስተቀር, ካል…)
- Preserve references (አንቀጽ, ሕግ, መመሪያ)

If removing any word changes the legal effect, **do not remove it**.

---

### 3.2 Sentence-Level Only

- Input is ONE sentence
- Output may be:
  - One shorter sentence, or
  - Two short sentences (if clarity improves)
- Do NOT merge multiple input sentences
- Do NOT create bullet points

---

## 4. Allowed Simplification Operations

The following operations are **explicitly allowed**.

### 4.1 Sentence Splitting

Long sentences may be split into two shorter sentences if:

- Each sentence remains grammatically complete
- Legal meaning is preserved

**Example**

Complex:
> በህግ መሠረት የተወሰነው ውሳኔ በሁሉም አካላት መከበር አለበት።

Simple:
> ይህ ውሳኔ በህግ መሠረት ነው። ሁሉም አካላት መከበር አለባቸው።

---

### 4.2 Archaic → Common Vocabulary Replacement

Formal or archaic legal words may be replaced with **commonly used Amharic equivalents**, as defined in the glossary.

**Allowed**
- ይገደዳል → አለበት
- ተፈጻሚነት → መስራት / መፈጸም (context-dependent)
- መሠረት → በ… ምክንያት / በ… መሠረት

Always follow the glossary.

---

### 4.3 Removing Redundant Legal Phrases

Boilerplate phrases that do not change legal meaning may be removed.

**Examples**
- “እንደተጠበቀ ሆኖ”
- “በማንኛውም ሁኔታ”

Only remove if:
- No condition is lost
- No obligation is weakened

---

### 4.4 Making the Agent Explicit

If a sentence is vague but the agent is clear from context, it may be made explicit.

**Example**

Complex:
> መክፈል አለበት።

Simple:
> ተጠቃሚው መክፈል አለበት።

---

## 5. Forbidden Operations (Critical)

The following are **NOT allowed under any circumstances**.

### 5.1 Do NOT Weaken Legal Force

- Do NOT change:
  - አለበት → ይገባል → ይችላል
- Do NOT soften mandatory language

---

### 5.2 Do NOT Remove Conditions

Never remove or rewrite away:

- ከ… በስተቀር
- ካል… በስተቀር
- ቢሆንም
- እስከሚ… ድረስ

If a condition exists, it MUST remain.

---

### 5.3 Do NOT Change Legal Roles or Entities

Do NOT modify:
- ከሳሽ
- ተከሳሽ
- ባለቤት
- Court names
- Institutions
- Article numbers

These are **unchangeable**.

---

### 5.4 Do NOT Explain or Interpret

Do NOT:
- Add explanations
- Define legal terms
- Summarize intent
- Add examples

Only rewrite.

---

## 6. Target Output Style

The simplified sentence should be:

- Shorter than the original
- Grammatically correct Amharic
- Written in everyday Amharic
- Neutral and factual
- Free of English or code-switching

Avoid overly conversational language.

---

## 7. Consistency Rules

- Similar structures must be simplified in similar ways
- Same legal phrase should map to the same plain phrase when possible
- Follow the glossary strictly

Consistency is more important than creativity.

---

## 8. Final Validation Checklist (Before Accepting a Pair)

Before a pair is added to the dataset, confirm:

- [ ] Legal meaning preserved
- [ ] No obligation removed
- [ ] No condition removed
- [ ] Roles unchanged
- [ ] Output is clearer than input
- [ ] Output is still legally precise

If any box fails, revise or discard the pair.

---

## 9. Scope Reminder

This dataset teaches the model **how to simplify legal sentences**, not how to understand law.

Legal grounding will be handled separately via external context.

---


