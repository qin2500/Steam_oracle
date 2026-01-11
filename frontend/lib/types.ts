export interface Assessment {
    criterion: string;
    status: 'met' | 'partial' | 'missing' | 'violated';
    evidence: string;
    category: string;
    confidence?: number;
}

export interface GameResult {
    id: string;
    name: string;
    score: number;
    reasoning: string;
    assessments: Assessment[];
    summary: string;
}

export interface Interpretation {
    // Define structure if known, otherwise Record<string, any>
    [key: string]: any;
}

export type SearchStatus = 'idle' | 'interpreting' | 'searching' | 'completed' | 'error';

export interface SearchState {
    status: SearchStatus;
    query: string;
    interpretation: Interpretation | null;
    results: GameResult[];
    error: string | null;
}

export type StreamEvent =
    | { type: 'optimization'; data: Interpretation }
    | { type: 'progress'; message: string }
    | { type: 'result'; data: GameResult[] }
    | { type: 'error'; message: string };
