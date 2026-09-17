use super::app::UiCommand;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TerminalSize {
    width: usize,
    height: usize,
}

fn compose_frame(contents: &str, size: TerminalSize) -> String {
    if size.width == 0 || size.height == 0 {
        return String::new();
    }
    let lines: Vec<String> = contents
        .lines()
        .map(|line| sanitize_and_clip(line, size.width))
        .collect();
    if lines.is_empty() {
        return String::new();
    }

    let visible: Vec<&str> = if lines.len() <= size.height {
        lines.iter().map(String::as_str).collect()
    } else if size.height == 1 {
        vec![lines[0].as_str()]
    } else if size.height == 2 {
        vec![lines[0].as_str(), lines[lines.len() - 1].as_str()]
    } else {
        let head_count = (size.height - 1) / 2;
        let tail_count = size.height - head_count - 1;
        let mut selected = Vec::with_capacity(size.height);
        selected.extend(lines[..head_count].iter().map(String::as_str));
        selected.push("... content clipped to terminal size ...");
        selected.extend(
            lines[lines.len() - tail_count..]
                .iter()
                .map(String::as_str),
        );
        selected
    };

    let mut frame = String::new();
    for (index, line) in visible.iter().enumerate() {
        frame.push_str(&sanitize_and_clip(line, size.width));
        if index + 1 < visible.len() {
            frame.push_str("\r\n");
        }
    }
    frame
}

fn sanitize_and_clip(line: &str, width: usize) -> String {
    let mut output = String::new();
    let mut columns = 0usize;
    for character in line.chars() {
        if columns >= width {
            break;
        }
        if character == '\t' {
            let spaces = 4usize.min(width - columns);
            for _ in 0..spaces {
                output.push(' ');
            }
            columns += spaces;
        } else if !character.is_control() {
            output.push(character);
            columns += 1;
        }
    }
    output
}

#[cfg(unix)]
mod platform {
    use std::io::{self, Read, Write};

    use super::{compose_frame, TerminalSize, UiCommand};

    const ESCAPE_SEQUENCE_TIMEOUT_MS: i32 = 100;
    const MAX_ESCAPE_SEQUENCE_BYTES: usize = 16;

    pub struct TerminalSession {
        stdin: io::Stdin,
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
            unsafe { libc::cfmakeraw(&mut raw) };
            raw.c_cc[libc::VMIN] = 1;
            raw.c_cc[libc::VTIME] = 0;
            if unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &raw) } != 0 {
                return Err(io::Error::last_os_error());
            }

            let mut session = Self {
                stdin: io::stdin(),
                stdout: io::stdout(),
                original,
            };
            if let Err(error) = session.stdout.write_all(b"\x1b[2J\x1b[H\x1b[?25l") {
                drop(session);
                return Err(error);
            }
            if let Err(error) = session.stdout.flush() {
                drop(session);
                return Err(error);
            }
            Ok(session)
        }

        pub fn draw(&mut self, contents: &str) -> io::Result<()> {
            let size = terminal_size()?;
            let frame = compose_frame(contents, size);
            self.stdout.write_all(b"\x1b[0m\x1b[2J\x1b[H")?;
            self.stdout.write_all(frame.as_bytes())?;
            self.stdout.flush()
        }

        pub fn read_command(&mut self, text_mode: bool) -> io::Result<UiCommand> {
            let first = self.read_byte()?;
            match first {
                3 => Ok(UiCommand::Quit),
                b'\r' | b'\n' => Ok(UiCommand::Enter),
                8 | 127 => Ok(UiCommand::Backspace),
                27 => self.read_escape_sequence(),
                b'q' if !text_mode => Ok(UiCommand::Quit),
                b'k' if !text_mode => Ok(UiCommand::Up),
                b'j' if !text_mode => Ok(UiCommand::Down),
                byte if (0x20..=0x7e).contains(&byte) => {
                    Ok(UiCommand::Character(byte as char))
                }
                byte if !byte.is_ascii() => self.read_utf8_character(byte),
                _ => Ok(UiCommand::None),
            }
        }

        fn read_byte(&mut self) -> io::Result<u8> {
            let mut byte = [0u8; 1];
            self.stdin.read_exact(&mut byte)?;
            Ok(byte[0])
        }

        fn read_byte_with_timeout(&mut self, timeout_ms: i32) -> io::Result<Option<u8>> {
            if stdin_ready(timeout_ms)? {
                self.read_byte().map(Some)
            } else {
                Ok(None)
            }
        }

        fn read_escape_sequence(&mut self) -> io::Result<UiCommand> {
            let Some(prefix) = self.read_byte_with_timeout(ESCAPE_SEQUENCE_TIMEOUT_MS)? else {
                return Ok(UiCommand::Back);
            };
            match prefix {
                b'[' => self.read_csi_sequence(),
                b'O' => match self.read_byte_with_timeout(ESCAPE_SEQUENCE_TIMEOUT_MS)? {
                    Some(b'A') => Ok(UiCommand::Up),
                    Some(b'B') => Ok(UiCommand::Down),
                    _ => Ok(UiCommand::None),
                },
                _ => Ok(UiCommand::Back),
            }
        }

        fn read_csi_sequence(&mut self) -> io::Result<UiCommand> {
            for _ in 0..MAX_ESCAPE_SEQUENCE_BYTES {
                let Some(byte) = self.read_byte_with_timeout(ESCAPE_SEQUENCE_TIMEOUT_MS)? else {
                    return Ok(UiCommand::None);
                };
                if (0x40..=0x7e).contains(&byte) {
                    return match byte {
                        b'A' => Ok(UiCommand::Up),
                        b'B' => Ok(UiCommand::Down),
                        _ => Ok(UiCommand::None),
                    };
                }
            }
            Ok(UiCommand::None)
        }

        fn read_utf8_character(&mut self, first: u8) -> io::Result<UiCommand> {
            let length = match first {
                0xc2..=0xdf => 2,
                0xe0..=0xef => 3,
                0xf0..=0xf4 => 4,
                _ => return Ok(UiCommand::None),
            };
            let mut bytes = [0u8; 4];
            bytes[0] = first;
            for byte in bytes.iter_mut().take(length).skip(1) {
                let Some(next) = self.read_byte_with_timeout(ESCAPE_SEQUENCE_TIMEOUT_MS)? else {
                    return Ok(UiCommand::None);
                };
                *byte = next;
            }
            let character = std::str::from_utf8(&bytes[..length])
                .ok()
                .and_then(|value| value.chars().next());
            Ok(character
                .map(UiCommand::Character)
                .unwrap_or(UiCommand::None))
        }
    }

    impl Drop for TerminalSession {
        fn drop(&mut self) {
            let _ = unsafe {
                libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &self.original)
            };
            let _ = self
                .stdout
                .write_all(b"\x1b[0m\x1b[?25h\x1b[2J\x1b[H");
            let _ = self.stdout.flush();
        }
    }

    fn stdin_ready(timeout_ms: i32) -> io::Result<bool> {
        let mut descriptor = libc::pollfd {
            fd: libc::STDIN_FILENO,
            events: libc::POLLIN,
            revents: 0,
        };
        loop {
            let result = unsafe { libc::poll(&mut descriptor, 1, timeout_ms) };
            if result >= 0 {
                return Ok(result > 0 && descriptor.revents & libc::POLLIN != 0);
            }
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
    }

    fn terminal_size() -> io::Result<TerminalSize> {
        let mut window = unsafe { std::mem::zeroed::<libc::winsize>() };
        if unsafe { libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut window) } != 0 {
            return Err(io::Error::last_os_error());
        }
        if window.ws_col == 0 || window.ws_row == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "terminal reported a zero-sized window",
            ));
        }
        Ok(TerminalSize {
            width: window.ws_col as usize,
            height: window.ws_row as usize,
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_clips_width_and_preserves_top_and_bottom_on_short_terminals() {
        let contents = "header\nsecond line is too long\nthird\nfourth\nfooter";
        let frame = compose_frame(
            contents,
            TerminalSize {
                width: 10,
                height: 4,
            },
        );
        let lines: Vec<&str> = frame.split("\r\n").collect();
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0], "header");
        assert_eq!(lines[1], "... conten");
        assert_eq!(lines[2], "fourth");
        assert_eq!(lines[3], "footer");
        assert!(lines.iter().all(|line| line.chars().count() <= 10));
    }

    #[test]
    fn frame_removes_control_sequences_from_content() {
        let frame = compose_frame(
            "safe\x1b[31mtext\nfooter",
            TerminalSize {
                width: 80,
                height: 24,
            },
        );
        assert!(!frame.contains('\x1b'));
        assert!(frame.contains("safe[31mtext"));
    }
}
