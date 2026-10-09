"""Entry point: refresh the CDP warehouse and show headline metrics."""

from src.cdp.pipeline import run_pipeline


def main():
    run_pipeline(num_customers=1200)


if __name__ == "__main__":
    main()