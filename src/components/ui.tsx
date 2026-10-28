import type { ReactNode } from "react";

export function Panel({
  title,
  right,
  children,
  className = "",
  bodyClassName = "p-3",
}: {
  title: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section className={`rounded-md border border-hud-border bg-hud-panel ${className}`}>
      <header className="flex items-center justify-between gap-2 border-b border-hud-border px-3 py-2">
        <h2 className="text-[11px] font-semibold tracking-[0.2em] text-hud-amber">{title}</h2>
        {right}
      </header>
      <div className={bodyClassName}>{children}</div>
    </section>
  );
}

export function Stat({ label, value, color }: { label: string; value: ReactNode; color?: string }) {
  return (
    <div className="rounded border border-hud-border bg-hud-panel2 px-3 py-2">
      <div className="text-[10px] tracking-[0.18em] text-hud-grey">{label}</div>
      <div className="mt-0.5 truncate text-lg font-semibold" style={{ color: color ?? "#f2f2f2" }}>
        {value}
      </div>
    </div>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  labels,
}: {
  value: T;
  options: readonly T[];
  onChange: (v: T) => void;
  labels?: Partial<Record<T, string>>;
}) {
  return (
    <div className="inline-flex overflow-hidden rounded border border-hud-border">
      {options.map((opt) => (
        <button
          key={opt}
          type="button"
          onClick={() => onChange(opt)}
          className={`px-2.5 py-1 text-[11px] tracking-[0.12em] transition-colors ${
            value === opt ? "bg-hud-amber text-hud-bg" : "bg-hud-panel2 text-hud-grey hover:text-hud-white"
          }`}
        >
          {(labels?.[opt] ?? opt).toUpperCase()}
        </button>
      ))}
    </div>
  );
}

export function Button({
  children,
  onClick,
  variant = "default",
  disabled,
  type = "button",
  className = "",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "default" | "primary" | "danger";
  disabled?: boolean;
  type?: "button" | "submit";
  className?: string;
}) {
  const styles = {
    default: "border-hud-border bg-hud-panel2 text-hud-white hover:border-hud-amber",
    primary: "border-hud-amber bg-hud-amber text-hud-bg hover:brightness-110",
    danger: "border-hud-red/60 bg-transparent text-hud-red hover:bg-hud-red/10",
  }[variant];
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`rounded border px-3 py-1.5 text-[11px] font-semibold tracking-[0.14em] transition disabled:cursor-not-allowed disabled:opacity-40 ${styles} ${className}`}
    >
      {children}
    </button>
  );
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[10px] tracking-[0.18em] text-hud-grey">{label}</span>
      {children}
      {hint ? <span className="mt-1 block text-[10px] text-hud-muted">{hint}</span> : null}
    </label>
  );
}
