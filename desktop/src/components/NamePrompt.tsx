import { useEffect, useRef, useState } from 'react';

interface Props {
    title: string;
    detail?: string;
    initial?: string;
    confirmLabel?: string;
    onConfirm: (name: string) => void;
    onCancel: () => void;
}

/**
 * Inline modal for naming a tag or person.
 *
 * Replaces `window.prompt`, which blocks the whole renderer, cannot be styled,
 * and looks nothing like the rest of the app.
 */
export function NamePrompt({
    title,
    detail,
    initial = '',
    confirmLabel = 'Create',
    onConfirm,
    onCancel,
}: Props) {
    const [value, setValue] = useState(initial);
    const input = useRef<HTMLInputElement>(null);

    useEffect(() => {
        input.current?.focus();
        input.current?.select();
    }, []);

    useEffect(() => {
        const onKey = (event: KeyboardEvent) => {
            if (event.key === 'Escape') onCancel();
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [onCancel]);

    return (
        <div className="modal-backdrop" onMouseDown={onCancel}>
            <form
                className="modal"
                onMouseDown={(event) => event.stopPropagation()}
                onSubmit={(event) => {
                    event.preventDefault();
                    const trimmed = value.trim();
                    if (trimmed) onConfirm(trimmed);
                }}
            >
                <h3>{title}</h3>
                {detail && <p className="muted">{detail}</p>}
                <input
                    ref={input}
                    value={value}
                    placeholder="name"
                    onChange={(event) => setValue(event.target.value)}
                />
                <div className="modal-actions">
                    <button type="button" onClick={onCancel}>
                        Cancel
                    </button>
                    <button type="submit" className="primary" disabled={!value.trim()}>
                        {confirmLabel}
                    </button>
                </div>
            </form>
        </div>
    );
}
