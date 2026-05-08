import os
import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_PROMPT = Path(__file__).resolve().parent.parent / "prompts" / "patra_gen_prompt.txt"

def load_prompt(prompt_file: str) -> str:
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def generate_email(context_emails: list[str], system_prompt: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"The list of the previous 5 emails are {context_emails}"}
        ]
    )
    return response.choices[0].message.content


def create_email_dataset(num_emails: int, seed_emails: list[str], context_window: int, system_prompt: str) -> list[str]:
    generated_emails = list(seed_emails)
    for _ in tqdm(range(num_emails), desc="Generating emails"):
        context_emails = generated_emails[-context_window:]
        new_email = generate_email(context_emails, system_prompt)
        generated_emails.append(new_email)
    return generated_emails


def save_dataset(emails: list[str], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\nEMAIL_END\n\n".join(emails))
    logging.info(f"Saved {len(emails)} emails to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate PATRA academic email corpus")
    parser.add_argument("--seed_file", required=True, help="Path to seed emails file (emails separated by EMAIL_END)")
    parser.add_argument("--output_file", required=True, help="Output path for generated dataset")
    parser.add_argument("--num_emails", type=int, default=1000, help="Number of emails to generate (default: 1000)")
    parser.add_argument("--context_window", type=int, default=7, help="Number of previous emails used as context (default: 7)")
    parser.add_argument("--prompt_file", default=str(DEFAULT_PROMPT), help=f"Path to system prompt file (default: {DEFAULT_PROMPT})")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    system_prompt = load_prompt(args.prompt_file)
    logging.info(f"Loaded system prompt from {args.prompt_file}")

    with open(args.seed_file, "r", encoding="utf-8") as f:
        raw = f.read()

    seed_emails = [e.strip() for e in raw.split("EMAIL_END") if e.strip()]
    logging.info(f"Loaded {len(seed_emails)} seed emails from {args.seed_file}")

    emails = create_email_dataset(
        num_emails=args.num_emails,
        seed_emails=seed_emails,
        context_window=args.context_window,
        system_prompt=system_prompt
    )

    save_dataset(emails, args.output_file)
    logging.info(f"Done. Generated {args.num_emails} new emails.")


if __name__ == "__main__":
    main()
