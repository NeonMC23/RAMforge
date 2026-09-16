use super::app::UiCommand;

#[cfg(unix)]
mod platform {
    use std::io::{self, Read, Write};

    use super::UiCommand;

    pub struct TerminalSession {
        stdout: io::Stdout,
        original: libc::termios,
    }

    impl TerminalSession {
        pub fn new() -> io::Result<Self> {
            if unsafe { libc::isatty(libc::STDIN_FILENO) } != 1
                || unsafe { libc::isatty(libc::STDOUT_FILENO) } != 1
            {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "ramforge-tui requires an interactive terminal",
                ));
            }

            let mut original = unsafe { std::mem::zeroed::<libc::termios>() };
            if unsafe { libc::tcgetattr(libc::STDIN_FILENO, &mut original) } != 0 {
                return Err(io::Error::last_os_error());
            }
            let mut raw = original;
            raw.c_iflag &= !(libc::BRKINT | libc::ICRNL | libc::INPCK | libc::ISTRIP | libc::IXON);
            raw.c_cflag |= libc::CS8;
            raw.c_lflag &= !(libc::ECHO | libc::ICANON | libc::IEXTEN | libc::ISIG);
            raw.c_cc[libc::VMIN] = 1;
            raw.c_cc[libc::VTIME] = 0;
            if unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSAFLUSH, &raw) } != 0 {
                return Err(io::Error::last_os_error());
            }

            let mut session = Self {
                stdout: io::stdout(),
                original,
            };
            session.stdout.write_all(b"\x1b[?1049h\x1b[?25l")?;
            session.stdout.flush()?;
            Ok(session)
        }

        pub fn draw(&mut self, contents: &str) -> io::Result<()> {
            self.stdout.write_all(b"\x1b[2J\x1b[H")?;
            self.stdout.write_all(contents.as_bytes())?;
            self.stdout.flush()
        }

        pub fn read_command(&mut self, text_mode: bool) -> io::Result<UiCommand> {
            let first = read_byte()?;
            match first {
                3 => Ok(UiCommand::Quit),
                b'\r' | b'\n' => Ok(UiCommand::Enter),
                8 | 127 => Ok(UiCommand::Backspace),
                27 => read_escape_sequence(),
                b'q' if !text_mode => Ok(UiCommand::Quit),
                b'k' if !text_mode => Ok(UiCommand::Up),
                b'j' if !text_mode => Ok(UiCommand::Down),
                byte if byte.is_ascii() => Ok(UiCommand::Character(byte as char)),
                byte => read_utf8_character(byte),
            }
        }
    }

    impl Drop for TerminalSession {
        fn drop(&mut self) {
            let _ = unsafe {
                libc::tcsetattr(
                    libc::STDIN_FILENO,
                    libc::TCSAFLUSH,
                    &self.original,
                )
            };
            let _ = self.stdout.write_all(b"\x1b[?25h\x1b[?1049l");
            let _ = self.stdout.flush();
        }
    }

    fn read_byte() -> io::Result<u8> {
        let mut byte = [0u8; 1];
        io::stdin().read_exact(&mut byte)?;
        Ok(byte[0])
    }

    fn read_escape_sequence() -> io::Result<UiCommand> {
        if !stdin_ready(25)? {
            return Ok(UiCommand::Back);
        }
        if read_byte()? != b'[' || !stdin_ready(25)? {
            return Ok(UiCommand::Back);
        }
        match read_byte()? {
            b'A' => Ok(UiCommand::Up),
            b'B' => Ok(UiCommand::Down),
            _ => Ok(UiCommand::None),
        }
    }

    fn stdin_ready(timeout_ms: i32) -> io::Result<bool> {
        let mut descriptor = libc::pollfd {
            fd: libc::STDIN_FILENO,
            events: libc::POLLIN,
            revents: 0,
        };
        let result = unsafe { libc::poll(&mut descriptor, 1, timeout_ms) };
        if result < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(result > 0 && descriptor.revents & libc::POLLIN != 0)
        }
    }

    fn read_utf8_character(first: u8) -> io::Result<UiCommand> {
        let length = match first {
            0xc2..=0xdf => 2,
            0xe0..=0xef => 3,
            0xf0..=0xf4 => 4,
            _ => return Ok(UiCommand::None),
        };
        let mut bytes = [0u8; 4];
        bytes[0] = first;
        io::stdin().read_exact(&mut bytes[1..length])?;
        let character = std::str::from_utf8(&bytes[..length])
            .ok()
            .and_then(|value| value.chars().next());
        Ok(character
            .map(UiCommand::Character)
            .unwrap_or(UiCommand::None))
    }
}

#[cfg(not(unix))]
mod platform {
    use std::io;

    use super::UiCommand;

    pub struct TerminalSession;

    impl TerminalSession {
        pub fn new() -> io::Result<Self> {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "ramforge-tui raw terminal input is currently supported on Unix terminals",
            ))
        }

        pub fn draw(&mut self, _contents: &str) -> io::Result<()> {
            Ok(())
        }

        pub fn read_command(&mut self, _text_mode: bool) -> io::Result<UiCommand> {
            Ok(UiCommand::Quit)
        }
    }
}

pub use platform::TerminalSession;
