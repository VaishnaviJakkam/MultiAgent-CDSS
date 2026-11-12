import { ArrowLeft } from "lucide-react";

type BackButtonProps = {
  onClick: () => void;
  label?: string;
  disabled?: boolean;
};

export default function BackButton({
  onClick,
  label = "Back",
  disabled = false,
}: BackButtonProps) {
  return (
    <button
      type="button"
      className="back-button"
      onClick={onClick}
      disabled={disabled}
    >
      <ArrowLeft size={17} aria-hidden="true" />
      {label}
    </button>
  );
}
