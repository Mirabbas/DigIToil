import argparse

from app import App


def main():
    parser = argparse.ArgumentParser(
        description='Person processing from video')
    parser.add_argument('-i', type=str, help='video path', default=0)
    args = parser.parse_args()

    app = App(args.i)
    app.run()


if __name__ == "__main__":
    main()
