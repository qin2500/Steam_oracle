import { SearchState, StreamEvent } from './types';

export async function searchGames(
    query: string,
    onUpdate: (update: Partial<SearchState>) => void
): Promise<void> {
    try {
        const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        const response = await fetch(`${API_URL}/api/v1/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query }),
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        if (!response.body) {
            throw new Error('No response body');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        onUpdate({ status: 'interpreting' });

        while (true) {
            const { done, value } = await reader.read();

            if (done) {
                break;
            }

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');

            // Process all complete lines
            buffer = lines.pop() || ''; // Keep the last incomplete part in buffer

            for (const line of lines) {
                if (!line.trim()) continue;

                try {
                    const event: StreamEvent = JSON.parse(line);

                    switch (event.type) {
                        case 'optimization':
                            onUpdate({
                                interpretation: event.data,
                                status: 'searching'
                            });
                            break;
                        case 'progress':
                            // Optional: could add a specific message field to state
                            break;
                        case 'result':
                            onUpdate({
                                results: event.data,
                                status: 'completed'
                            });
                            break;
                        case 'error':
                            onUpdate({
                                error: event.message,
                                status: 'error'
                            });
                            break;
                    }
                } catch (e) {
                    console.warn('Failed to parse stream line:', line, e);
                }
            }
        }
    } catch (error) {
        onUpdate({
            error: error instanceof Error ? error.message : 'Unknown error',
            status: 'error'
        });
    }
}
