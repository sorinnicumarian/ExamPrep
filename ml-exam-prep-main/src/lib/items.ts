import type { Question } from '../types'

// A "study item" is the unit tracked by spaced repetition:
//  - for True/False questions, each sub-statement is its own item
//  - for every other type, the whole question is one item
export interface StudyItem {
  itemId: string
  question: Question
  subId?: string
}

export function itemsForQuestion(q: Question): StudyItem[] {
  if (q.type === 'true-false-motivated' && q.subStatements)
    return q.subStatements.map((s) => ({ itemId: s.id, question: q, subId: s.id }))
  return [{ itemId: q.id, question: q }]
}

export function allItems(questions: Question[]): StudyItem[] {
  return questions.flatMap(itemsForQuestion)
}
