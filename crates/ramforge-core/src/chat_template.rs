//! Conservative interpreter for GGUF `tokenizer.chat_template` metadata.
//!
//! GGUF stores chat templates as Jinja-style source. Running a full Jinja
//! engine would add a dependency and a large behavior surface, so this module
//! implements the subset needed to render ordinary chat conversations:
//!
//! - `{% if %}` / `{% elif %}` / `{% else %}` / `{% endif %}`
//! - `{% for x in list %}` / `{% endfor %}` with `loop.first`, `loop.last`,
//!   `loop.index0`, `loop.index`, and `loop.length`
//! - `{% set name = expression %}`
//! - string, number, boolean, null, list, and object values
//! - indexing, member access, `+`, comparisons, `and`, `or`, `not`
//! - `is defined`
//! - the `tojson` filter
//! - Jinja whitespace-trimming markers (`{%-`, `-%}`, `{{-`, `-}}`)
//!
//! Anything outside that subset fails with an explicit error. A template is
//! never silently ignored or replaced by a model-specific prompt format.

use std::collections::BTreeMap;

use thiserror::Error;

use crate::model::GgufModel;

/// A single chat message supplied to a template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ChatTemplateError {
    #[error("model does not declare tokenizer.chat_template")]
    Missing,
    #[error("invalid tokenizer.chat_template: {0}")]
    Parse(String),
    #[error("unsupported tokenizer.chat_template construct: {0}")]
    Unsupported(String),
    #[error("failed to render tokenizer.chat_template: {0}")]
    Render(String),
}

/// A parsed chat template ready to render message lists.
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    source: String,
    nodes: Vec<Node>,
}

impl ChatTemplate {
    /// Read and parse `tokenizer.chat_template` from a GGUF model.
    pub fn from_gguf(model: &GgufModel) -> Result<Self, ChatTemplateError> {
        let source = model
            .get_metadata("tokenizer.chat_template")
            .and_then(|value| value.as_string())
            .ok_or(ChatTemplateError::Missing)?;
        Self::parse(source)
    }

    /// Parse a chat template from its Jinja-style source.
    pub fn parse(source: &str) -> Result<Self, ChatTemplateError> {
        let tokens = tokenize_template(source)?;
        let mut parser = TemplateParser {
            tokens,
            position: 0,
        };
        let (nodes, terminator) = parser.parse_nodes(&[])?;
        if let Some(terminator) = terminator {
            return Err(ChatTemplateError::Parse(format!(
                "unexpected template terminator '{terminator}'"
            )));
        }
        Ok(Self {
            source: source.to_string(),
            nodes,
        })
    }

    /// Original template source, useful for diagnostics.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Render a message list and optionally append an assistant generation cue.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String, ChatTemplateError> {
        let message_values = messages
            .iter()
            .map(|message| {
                let mut object = BTreeMap::new();
                object.insert("role".to_string(), Value::String(message.role.clone()));
                object.insert(
                    "content".to_string(),
                    Value::String(message.content.clone()),
                );
                Value::Object(object)
            })
            .collect();

        let mut scope = Scope::new();
        scope.set("messages", Value::List(message_values));
        // An empty list is falsy, matching the common `{% if tools %}` guard
        // without inventing tool definitions that the caller did not supply.
        scope.set("tools", Value::List(Vec::new()));
        scope.set("add_generation_prompt", Value::Bool(add_generation_prompt));

        let mut output = String::new();
        render_nodes(&self.nodes, &mut scope, &mut output)?;
        Ok(output)
    }
}

#[derive(Debug, Clone)]
enum Node {
    Text(String),
    Output(Expression),
    If {
        branches: Vec<(Expression, Vec<Node>)>,
        otherwise: Option<Vec<Node>>,
    },
    For {
        variable: String,
        iterable: Expression,
        body: Vec<Node>,
    },
    Set {
        variable: String,
        value: Expression,
    },
}

#[derive(Debug, Clone)]
enum Expression {
    Literal(Value),
    Variable(String),
    Member(Box<Expression>, String),
    Index(Box<Expression>, Box<Expression>),
    UnaryNot(Box<Expression>),
    Binary {
        operator: BinaryOperator,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    IsDefined(Box<Expression>),
    Filter {
        value: Box<Expression>,
        name: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinaryOperator {
    Add,
    Subtract,
    Equal,
    NotEqual,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq)]
enum Value {
    Undefined,
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    List(Vec<Value>),
    Object(BTreeMap<String, Value>),
}

impl Value {
    fn truthy(&self) -> bool {
        match self {
            Self::Undefined | Self::Null => false,
            Self::Bool(value) => *value,
            Self::Number(value) => *value != 0.0,
            Self::String(value) => !value.is_empty(),
            Self::List(values) => !values.is_empty(),
            Self::Object(values) => !values.is_empty(),
        }
    }

    fn render(&self) -> String {
        match self {
            Self::Undefined => String::new(),
            Self::Null => "none".to_string(),
            Self::Bool(value) => value.to_string(),
            Self::Number(value) => {
                if value.fract() == 0.0 && value.is_finite() {
                    format!("{}", *value as i64)
                } else {
                    value.to_string()
                }
            }
            Self::String(value) => value.clone(),
            Self::List(values) => {
                let rendered: Vec<String> = values.iter().map(Value::render).collect();
                format!("[{}]", rendered.join(", "))
            }
            Self::Object(values) => {
                let rendered: Vec<String> = values
                    .iter()
                    .map(|(key, value)| format!("{key}: {}", value.render()))
                    .collect();
                format!("{{{}}}", rendered.join(", "))
            }
        }
    }

    fn to_json(&self) -> String {
        match self {
            Self::Undefined | Self::Null => "null".to_string(),
            Self::Bool(value) => value.to_string(),
            Self::Number(value) => {
                if value.fract() == 0.0 && value.is_finite() {
                    format!("{}", *value as i64)
                } else {
                    value.to_string()
                }
            }
            Self::String(value) => json_string(value),
            Self::List(values) => {
                let rendered: Vec<String> = values.iter().map(Value::to_json).collect();
                format!("[{}]", rendered.join(","))
            }
            Self::Object(values) => {
                let rendered: Vec<String> = values
                    .iter()
                    .map(|(key, value)| format!("{}:{}", json_string(key), value.to_json()))
                    .collect();
                format!("{{{}}}", rendered.join(","))
            }
        }
    }
}

fn json_string(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character.is_control() => {
                output.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => output.push(character),
        }
    }
    output.push('"');
    output
}

#[derive(Debug, Clone)]
enum TemplateToken {
    Text(String),
    Expression(String),
    Statement(String),
}

/// Find a tag terminator while ignoring markers inside quoted strings. Chat
/// templates may embed `}}` inside a JSON example in an output expression.
fn find_tag_end(source: &str, marker: &str) -> Option<usize> {
    let mut quote = None;
    let mut escaped = false;
    let mut cursor = 0;
    while cursor < source.len() {
        let character = source[cursor..].chars().next()?;
        if escaped {
            escaped = false;
            cursor += character.len_utf8();
            continue;
        }
        if let Some(active_quote) = quote {
            if character == '\\' {
                escaped = true;
            } else if character == active_quote {
                quote = None;
            }
        } else if character == '\'' || character == '"' {
            quote = Some(character);
        } else if source[cursor..].starts_with(marker) {
            return Some(cursor);
        }
        cursor += character.len_utf8();
    }
    None
}

fn tokenize_template(source: &str) -> Result<Vec<TemplateToken>, ChatTemplateError> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut cursor = 0;
    let mut text_start = 0;
    let mut trim_next_text = false;

    while cursor + 1 < bytes.len() {
        let opening = if &bytes[cursor..cursor + 2] == b"{{" {
            Some((2usize, "}}"))
        } else if &bytes[cursor..cursor + 2] == b"{%" {
            Some((2usize, "%}"))
        } else if &bytes[cursor..cursor + 2] == b"{#" {
            Some((2usize, "#}"))
        } else {
            None
        };

        let Some((open_len, close_marker)) = opening else {
            cursor += source[cursor..]
                .chars()
                .next()
                .map(char::len_utf8)
                .unwrap_or(1);
            continue;
        };

        let mut content_start = cursor + open_len;
        let trim_before = bytes.get(content_start) == Some(&b'-');
        if trim_before {
            content_start += 1;
        }
        let Some(relative_close) = find_tag_end(&source[content_start..], close_marker) else {
            return Err(ChatTemplateError::Parse(format!(
                "unterminated template tag at byte {cursor}"
            )));
        };
        let close_start = content_start + relative_close;
        let mut content_end = close_start;
        let trim_after = content_end > content_start && bytes[content_end - 1] == b'-';
        if trim_after {
            content_end -= 1;
        }

        let mut text = &source[text_start..cursor];
        if trim_next_text {
            text = text.trim_start();
        }
        if trim_before {
            text = text.trim_end();
        }
        if !text.is_empty() {
            tokens.push(TemplateToken::Text(text.to_string()));
        }

        let content = source[content_start..content_end].trim();
        if close_marker != "#}" {
            if content.is_empty() {
                return Err(ChatTemplateError::Parse(format!(
                    "empty template tag at byte {cursor}"
                )));
            }
            match close_marker {
                "}}" => tokens.push(TemplateToken::Expression(content.to_string())),
                "%}" => tokens.push(TemplateToken::Statement(content.to_string())),
                _ => unreachable!(),
            }
        }

        trim_next_text = trim_after;
        cursor = close_start + close_marker.len();
        text_start = cursor;
    }

    let mut tail = &source[text_start..];
    if trim_next_text {
        tail = tail.trim_start();
    }
    if !tail.is_empty() {
        tokens.push(TemplateToken::Text(tail.to_string()));
    }
    Ok(tokens)
}

struct TemplateParser {
    tokens: Vec<TemplateToken>,
    position: usize,
}

impl TemplateParser {
    fn parse_nodes(
        &mut self,
        terminators: &[&str],
    ) -> Result<(Vec<Node>, Option<String>), ChatTemplateError> {
        let mut nodes = Vec::new();
        while self.position < self.tokens.len() {
            match self.tokens[self.position].clone() {
                TemplateToken::Text(text) => {
                    self.position += 1;
                    nodes.push(Node::Text(text));
                }
                TemplateToken::Expression(source) => {
                    self.position += 1;
                    nodes.push(Node::Output(parse_expression(&source)?));
                }
                TemplateToken::Statement(statement) => {
                    let keyword = statement
                        .split_whitespace()
                        .next()
                        .unwrap_or_default()
                        .to_string();
                    if terminators.iter().any(|terminator| *terminator == keyword) {
                        return Ok((nodes, Some(statement)));
                    }
                    match keyword.as_str() {
                        "if" => {
                            let condition = statement
                                .strip_prefix("if")
                                .ok_or_else(|| {
                                    ChatTemplateError::Parse("missing if condition".to_string())
                                })?
                                .trim();
                            if condition.is_empty() {
                                return Err(ChatTemplateError::Parse(
                                    "empty if condition".to_string(),
                                ));
                            }
                            self.position += 1;
                            nodes.push(self.parse_if(condition)?);
                        }
                        "for" => {
                            self.position += 1;
                            nodes.push(self.parse_for(&statement)?);
                        }
                        "set" => {
                            self.position += 1;
                            nodes.push(parse_set(&statement)?);
                        }
                        "elif" | "else" | "endif" | "endfor" => {
                            return Ok((nodes, Some(statement)));
                        }
                        "" => {
                            return Err(ChatTemplateError::Parse(
                                "empty template statement".to_string(),
                            ));
                        }
                        unsupported => {
                            return Err(ChatTemplateError::Unsupported(format!(
                                "statement '{unsupported}'"
                            )));
                        }
                    }
                }
            }
        }
        Ok((nodes, None))
    }

    fn parse_if(&mut self, condition: &str) -> Result<Node, ChatTemplateError> {
        let condition = parse_expression(condition)?;
        let (body, mut terminator) = self.parse_nodes(&["elif", "else", "endif"])?;
        let mut branches = vec![(condition, body)];
        let mut otherwise = None;

        loop {
            let current = terminator
                .take()
                .ok_or_else(|| ChatTemplateError::Parse("unterminated if statement".to_string()))?;
            let keyword = current.split_whitespace().next().unwrap_or_default();
            match keyword {
                "elif" => {
                    let condition = current
                        .strip_prefix("elif")
                        .ok_or_else(|| {
                            ChatTemplateError::Parse("missing elif condition".to_string())
                        })?
                        .trim();
                    if condition.is_empty() {
                        return Err(ChatTemplateError::Parse("empty elif condition".to_string()));
                    }
                    self.position += 1;
                    let (body, next) = self.parse_nodes(&["elif", "else", "endif"])?;
                    branches.push((parse_expression(condition)?, body));
                    terminator = next;
                }
                "else" => {
                    self.position += 1;
                    let (body, next) = self.parse_nodes(&["endif"])?;
                    otherwise = Some(body);
                    terminator = next;
                }
                "endif" => {
                    self.position += 1;
                    break;
                }
                _ => {
                    return Err(ChatTemplateError::Parse(format!(
                        "unexpected terminator '{current}'"
                    )));
                }
            }
        }

        Ok(Node::If {
            branches,
            otherwise,
        })
    }

    fn parse_for(&mut self, statement: &str) -> Result<Node, ChatTemplateError> {
        let rest = statement
            .strip_prefix("for")
            .ok_or_else(|| ChatTemplateError::Parse("missing for loop".to_string()))?
            .trim();
        let (variable, iterable) = rest.split_once(" in ").ok_or_else(|| {
            ChatTemplateError::Parse("for loop must use 'for name in expression'".to_string())
        })?;
        let variable = variable.trim();
        if variable.is_empty() || variable.contains(char::is_whitespace) {
            return Err(ChatTemplateError::Parse(
                "for loop variable must be a single identifier".to_string(),
            ));
        }
        let (body, terminator) = self.parse_nodes(&["endfor"])?;
        let terminator = terminator
            .ok_or_else(|| ChatTemplateError::Parse("unterminated for statement".to_string()))?;
        if terminator.split_whitespace().next() != Some("endfor") {
            return Err(ChatTemplateError::Parse(format!(
                "unexpected for terminator '{terminator}'"
            )));
        }
        self.position += 1;
        Ok(Node::For {
            variable: variable.to_string(),
            iterable: parse_expression(iterable.trim())?,
            body,
        })
    }
}

fn parse_set(statement: &str) -> Result<Node, ChatTemplateError> {
    let rest = statement
        .strip_prefix("set")
        .ok_or_else(|| ChatTemplateError::Parse("missing set statement".to_string()))?
        .trim();
    let (variable, expression) = rest.split_once('=').ok_or_else(|| {
        ChatTemplateError::Parse("set statement must use 'set name = expression'".to_string())
    })?;
    let variable = variable.trim();
    if variable.is_empty() || variable.contains(char::is_whitespace) {
        return Err(ChatTemplateError::Parse(
            "set variable must be a single identifier".to_string(),
        ));
    }
    Ok(Node::Set {
        variable: variable.to_string(),
        value: parse_expression(expression.trim())?,
    })
}

#[derive(Debug, Clone, PartialEq)]
enum ExpressionToken {
    Identifier(String),
    Number(f64),
    String(String),
    Operator(String),
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    Dot,
    Pipe,
}

fn tokenize_expression(source: &str) -> Result<Vec<ExpressionToken>, ChatTemplateError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = source.chars().collect();
    let mut position = 0;

    while position < chars.len() {
        let character = chars[position];
        if character.is_whitespace() {
            position += 1;
            continue;
        }
        if character == '\'' || character == '"' {
            let quote = character;
            position += 1;
            let mut value = String::new();
            let mut terminated = false;
            while position < chars.len() {
                let character = chars[position];
                position += 1;
                if character == quote {
                    terminated = true;
                    break;
                }
                if character == '\\' {
                    let escaped = chars.get(position).copied().ok_or_else(|| {
                        ChatTemplateError::Parse("trailing escape in string literal".to_string())
                    })?;
                    position += 1;
                    value.push(match escaped {
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        '\\' => '\\',
                        '\'' => '\'',
                        '"' => '"',
                        other => other,
                    });
                } else {
                    value.push(character);
                }
            }
            if !terminated {
                return Err(ChatTemplateError::Parse(
                    "unterminated string literal".to_string(),
                ));
            }
            tokens.push(ExpressionToken::String(value));
            continue;
        }
        if character.is_ascii_digit() {
            let start = position;
            while position < chars.len()
                && (chars[position].is_ascii_digit() || chars[position] == '.')
            {
                position += 1;
            }
            let text: String = chars[start..position].iter().collect();
            let number = text.parse::<f64>().map_err(|_| {
                ChatTemplateError::Parse(format!("invalid number literal '{text}'"))
            })?;
            tokens.push(ExpressionToken::Number(number));
            continue;
        }
        if character.is_alphabetic() || character == '_' {
            let start = position;
            while position < chars.len()
                && (chars[position].is_alphanumeric() || chars[position] == '_')
            {
                position += 1;
            }
            let identifier: String = chars[start..position].iter().collect();
            tokens.push(ExpressionToken::Identifier(identifier));
            continue;
        }

        let next = chars.get(position + 1).copied();
        let two_character = next.map(|next| format!("{character}{next}"));
        if let Some("==" | "!=" | ">=" | "<=") = two_character.as_deref() {
            tokens.push(ExpressionToken::Operator(
                two_character.expect("matched two-character operator"),
            ));
            position += 2;
            continue;
        }
        match character {
            '+' | '-' | '>' | '<' => {
                tokens.push(ExpressionToken::Operator(character.to_string()));
                position += 1;
            }
            '(' => {
                tokens.push(ExpressionToken::LeftParen);
                position += 1;
            }
            ')' => {
                tokens.push(ExpressionToken::RightParen);
                position += 1;
            }
            '[' => {
                tokens.push(ExpressionToken::LeftBracket);
                position += 1;
            }
            ']' => {
                tokens.push(ExpressionToken::RightBracket);
                position += 1;
            }
            '.' => {
                tokens.push(ExpressionToken::Dot);
                position += 1;
            }
            '|' => {
                tokens.push(ExpressionToken::Pipe);
                position += 1;
            }
            ',' => {
                // Commas are only useful for collection literals, which are
                // intentionally outside this subset. Reject them clearly.
                return Err(ChatTemplateError::Unsupported(
                    "comma-separated expression".to_string(),
                ));
            }
            unsupported => {
                return Err(ChatTemplateError::Unsupported(format!(
                    "operator '{unsupported}'"
                )));
            }
        }
    }

    Ok(tokens)
}

fn parse_expression(source: &str) -> Result<Expression, ChatTemplateError> {
    let tokens = tokenize_expression(source)
        .map_err(|error| ChatTemplateError::Parse(format!("in expression {source:?}: {error}")))?;
    let mut parser = ExpressionParser {
        tokens,
        position: 0,
    };
    let expression = parser.parse_or()?;
    if parser.position != parser.tokens.len() {
        return Err(ChatTemplateError::Parse(format!(
            "unexpected expression token at position {}",
            parser.position
        )));
    }
    Ok(expression)
}

struct ExpressionParser {
    tokens: Vec<ExpressionToken>,
    position: usize,
}

impl ExpressionParser {
    fn peek(&self) -> Option<&ExpressionToken> {
        self.tokens.get(self.position)
    }

    fn advance(&mut self) -> Option<ExpressionToken> {
        let token = self.tokens.get(self.position).cloned();
        if token.is_some() {
            self.position += 1;
        }
        token
    }

    fn keyword(&self, keyword: &str) -> bool {
        matches!(
            self.peek(),
            Some(ExpressionToken::Identifier(identifier)) if identifier == keyword
        )
    }

    fn consume_keyword(&mut self, keyword: &str) -> bool {
        if self.keyword(keyword) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn parse_or(&mut self) -> Result<Expression, ChatTemplateError> {
        let mut expression = self.parse_and()?;
        while self.consume_keyword("or") {
            let right = self.parse_and()?;
            expression = Expression::Binary {
                operator: BinaryOperator::Or,
                left: Box::new(expression),
                right: Box::new(right),
            };
        }
        Ok(expression)
    }

    fn parse_and(&mut self) -> Result<Expression, ChatTemplateError> {
        let mut expression = self.parse_not()?;
        while self.consume_keyword("and") {
            let right = self.parse_not()?;
            expression = Expression::Binary {
                operator: BinaryOperator::And,
                left: Box::new(expression),
                right: Box::new(right),
            };
        }
        Ok(expression)
    }

    fn parse_not(&mut self) -> Result<Expression, ChatTemplateError> {
        if self.consume_keyword("not") {
            Ok(Expression::UnaryNot(Box::new(self.parse_not()?)))
        } else {
            self.parse_comparison()
        }
    }

    fn parse_comparison(&mut self) -> Result<Expression, ChatTemplateError> {
        let mut expression = self.parse_concat()?;
        if self.consume_keyword("is") {
            if self.consume_keyword("defined") {
                return Ok(Expression::IsDefined(Box::new(expression)));
            }
            return Err(ChatTemplateError::Unsupported(
                "only the 'is defined' test is supported".to_string(),
            ));
        }
        let operator = match self.peek() {
            Some(ExpressionToken::Operator(operator)) => match operator.as_str() {
                "==" => Some(BinaryOperator::Equal),
                "!=" => Some(BinaryOperator::NotEqual),
                "<" => Some(BinaryOperator::Less),
                "<=" => Some(BinaryOperator::LessOrEqual),
                ">" => Some(BinaryOperator::Greater),
                ">=" => Some(BinaryOperator::GreaterOrEqual),
                _ => None,
            },
            _ => None,
        };
        if let Some(operator) = operator {
            self.position += 1;
            let right = self.parse_concat()?;
            expression = Expression::Binary {
                operator,
                left: Box::new(expression),
                right: Box::new(right),
            };
        }
        Ok(expression)
    }

    fn parse_concat(&mut self) -> Result<Expression, ChatTemplateError> {
        let mut expression = self.parse_postfix()?;
        loop {
            let operator = match self.peek() {
                Some(ExpressionToken::Operator(operator)) if operator == "+" => BinaryOperator::Add,
                Some(ExpressionToken::Operator(operator)) if operator == "-" => {
                    BinaryOperator::Subtract
                }
                _ => break,
            };
            self.position += 1;
            let right = self.parse_postfix()?;
            expression = Expression::Binary {
                operator,
                left: Box::new(expression),
                right: Box::new(right),
            };
        }
        Ok(expression)
    }

    fn parse_postfix(&mut self) -> Result<Expression, ChatTemplateError> {
        let mut expression = self.parse_primary()?;
        loop {
            match self.peek() {
                Some(ExpressionToken::Dot) => {
                    self.position += 1;
                    let member = match self.advance() {
                        Some(ExpressionToken::Identifier(identifier)) => identifier,
                        Some(ExpressionToken::Number(number)) => number.to_string(),
                        _ => {
                            return Err(ChatTemplateError::Parse(
                                "member access requires a name".to_string(),
                            ));
                        }
                    };
                    expression = Expression::Member(Box::new(expression), member);
                }
                Some(ExpressionToken::LeftBracket) => {
                    self.position += 1;
                    let index = self.parse_or()?;
                    if !matches!(self.advance(), Some(ExpressionToken::RightBracket)) {
                        return Err(ChatTemplateError::Parse(
                            "missing closing bracket".to_string(),
                        ));
                    }
                    expression = Expression::Index(Box::new(expression), Box::new(index));
                }
                Some(ExpressionToken::Pipe) => {
                    self.position += 1;
                    let filter = match self.advance() {
                        Some(ExpressionToken::Identifier(identifier)) => identifier,
                        _ => {
                            return Err(ChatTemplateError::Parse(
                                "filter name must be an identifier".to_string(),
                            ));
                        }
                    };
                    expression = Expression::Filter {
                        value: Box::new(expression),
                        name: filter,
                    };
                }
                _ => break,
            }
        }
        Ok(expression)
    }

    fn parse_primary(&mut self) -> Result<Expression, ChatTemplateError> {
        match self.advance() {
            Some(ExpressionToken::String(value)) => Ok(Expression::Literal(Value::String(value))),
            Some(ExpressionToken::Number(value)) => Ok(Expression::Literal(Value::Number(value))),
            Some(ExpressionToken::Identifier(identifier)) => match identifier.as_str() {
                "true" | "True" => Ok(Expression::Literal(Value::Bool(true))),
                "false" | "False" => Ok(Expression::Literal(Value::Bool(false))),
                "none" | "None" | "null" => Ok(Expression::Literal(Value::Null)),
                _ => Ok(Expression::Variable(identifier)),
            },
            Some(ExpressionToken::LeftParen) => {
                let expression = self.parse_or()?;
                if !matches!(self.advance(), Some(ExpressionToken::RightParen)) {
                    return Err(ChatTemplateError::Parse(
                        "missing closing parenthesis".to_string(),
                    ));
                }
                Ok(expression)
            }
            Some(ExpressionToken::LeftBracket) => {
                if matches!(self.peek(), Some(ExpressionToken::RightBracket)) {
                    self.position += 1;
                    Ok(Expression::Literal(Value::List(Vec::new())))
                } else {
                    Err(ChatTemplateError::Unsupported(
                        "non-empty list literal".to_string(),
                    ))
                }
            }
            token => Err(ChatTemplateError::Parse(format!(
                "expected expression, found {token:?}"
            ))),
        }
    }
}

struct Scope {
    frames: Vec<BTreeMap<String, Value>>,
}

impl Scope {
    fn new() -> Self {
        Self {
            frames: vec![BTreeMap::new()],
        }
    }

    fn get(&self, name: &str) -> Value {
        for frame in self.frames.iter().rev() {
            if let Some(value) = frame.get(name) {
                return value.clone();
            }
        }
        Value::Undefined
    }

    fn set(&mut self, name: &str, value: Value) {
        if let Some(frame) = self.frames.last_mut() {
            frame.insert(name.to_string(), value);
        }
    }

    fn push(&mut self) {
        self.frames.push(BTreeMap::new());
    }

    fn pop(&mut self) {
        self.frames.pop();
    }
}

fn render_nodes(
    nodes: &[Node],
    scope: &mut Scope,
    output: &mut String,
) -> Result<(), ChatTemplateError> {
    for node in nodes {
        match node {
            Node::Text(text) => output.push_str(text),
            Node::Output(expression) => output.push_str(&evaluate(expression, scope)?.render()),
            Node::If {
                branches,
                otherwise,
            } => {
                let mut matched = false;
                for (condition, body) in branches {
                    if evaluate(condition, scope)?.truthy() {
                        render_nodes(body, scope, output)?;
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    if let Some(body) = otherwise {
                        render_nodes(body, scope, output)?;
                    }
                }
            }
            Node::For {
                variable,
                iterable,
                body,
            } => {
                let values = match evaluate(iterable, scope)? {
                    Value::List(values) => values,
                    Value::Undefined | Value::Null => Vec::new(),
                    other => {
                        return Err(ChatTemplateError::Render(format!(
                            "cannot iterate over {}",
                            other.render()
                        )));
                    }
                };
                let length = values.len();
                for (index, value) in values.into_iter().enumerate() {
                    let mut loop_object = BTreeMap::new();
                    loop_object.insert("first".to_string(), Value::Bool(index == 0));
                    loop_object.insert("last".to_string(), Value::Bool(index + 1 == length));
                    loop_object.insert("index0".to_string(), Value::Number(index as f64));
                    loop_object.insert("index".to_string(), Value::Number((index + 1) as f64));
                    loop_object.insert("length".to_string(), Value::Number(length as f64));
                    scope.push();
                    scope.set(variable, value);
                    scope.set("loop", Value::Object(loop_object));
                    let result = render_nodes(body, scope, output);
                    scope.pop();
                    result?;
                }
            }
            Node::Set { variable, value } => {
                let value = evaluate(value, scope)?;
                scope.set(variable, value);
            }
        }
    }
    Ok(())
}

fn evaluate(expression: &Expression, scope: &Scope) -> Result<Value, ChatTemplateError> {
    match expression {
        Expression::Literal(value) => Ok(value.clone()),
        Expression::Variable(name) => Ok(scope.get(name)),
        Expression::Member(value, member) => {
            let value = evaluate(value, scope)?;
            Ok(match value {
                Value::Object(object) => object.get(member).cloned().unwrap_or(Value::Undefined),
                Value::Undefined => Value::Undefined,
                _ => Value::Undefined,
            })
        }
        Expression::Index(value, index) => {
            let value = evaluate(value, scope)?;
            let index = evaluate(index, scope)?;
            Ok(match (value, index) {
                (Value::List(values), Value::Number(index)) => values
                    .get(index as isize as usize)
                    .cloned()
                    .unwrap_or(Value::Undefined),
                (Value::Object(object), Value::String(index)) => {
                    object.get(&index).cloned().unwrap_or(Value::Undefined)
                }
                (Value::Undefined, _) => Value::Undefined,
                _ => Value::Undefined,
            })
        }
        Expression::UnaryNot(value) => Ok(Value::Bool(!evaluate(value, scope)?.truthy())),
        Expression::Binary {
            operator,
            left,
            right,
        } => match operator {
            BinaryOperator::And => {
                let left = evaluate(left, scope)?;
                if !left.truthy() {
                    Ok(left)
                } else {
                    evaluate(right, scope)
                }
            }
            BinaryOperator::Or => {
                let left = evaluate(left, scope)?;
                if left.truthy() {
                    Ok(left)
                } else {
                    evaluate(right, scope)
                }
            }
            BinaryOperator::Add => {
                let left = evaluate(left, scope)?;
                let right = evaluate(right, scope)?;
                match (left, right) {
                    (Value::String(left), Value::String(right)) => {
                        Ok(Value::String(format!("{left}{right}")))
                    }
                    (Value::Number(left), Value::Number(right)) => Ok(Value::Number(left + right)),
                    (left, right) => Err(ChatTemplateError::Render(format!(
                        "cannot add {} and {}",
                        left.render(),
                        right.render()
                    ))),
                }
            }
            BinaryOperator::Subtract => {
                let left = evaluate(left, scope)?;
                let right = evaluate(right, scope)?;
                match (left, right) {
                    (Value::Number(left), Value::Number(right)) => Ok(Value::Number(left - right)),
                    (left, right) => Err(ChatTemplateError::Render(format!(
                        "cannot subtract {} from {}",
                        right.render(),
                        left.render()
                    ))),
                }
            }
            BinaryOperator::Equal | BinaryOperator::NotEqual => {
                let equal = evaluate(left, scope)? == evaluate(right, scope)?;
                let equal = if *operator == BinaryOperator::NotEqual {
                    !equal
                } else {
                    equal
                };
                Ok(Value::Bool(equal))
            }
            BinaryOperator::Less
            | BinaryOperator::LessOrEqual
            | BinaryOperator::Greater
            | BinaryOperator::GreaterOrEqual => {
                let left = evaluate(left, scope)?;
                let right = evaluate(right, scope)?;
                let (Value::Number(left), Value::Number(right)) = (left, right) else {
                    return Ok(Value::Bool(false));
                };
                Ok(Value::Bool(match operator {
                    BinaryOperator::Less => left < right,
                    BinaryOperator::LessOrEqual => left <= right,
                    BinaryOperator::Greater => left > right,
                    BinaryOperator::GreaterOrEqual => left >= right,
                    _ => unreachable!(),
                }))
            }
        },
        Expression::IsDefined(value) => Ok(Value::Bool(!matches!(
            evaluate(value, scope)?,
            Value::Undefined
        ))),
        Expression::Filter { value, name } => {
            let value = evaluate(value, scope)?;
            match name.as_str() {
                "tojson" => Ok(Value::String(value.to_json())),
                unsupported => Err(ChatTemplateError::Unsupported(format!(
                    "filter '{unsupported}'"
                ))),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(source: &str, messages: &[ChatMessage], add_generation_prompt: bool) -> String {
        ChatTemplate::parse(source)
            .unwrap()
            .render(messages, add_generation_prompt)
            .unwrap()
    }

    #[test]
    fn renders_messages_and_generation_prompt() {
        let template = concat!(
            "{%- for message in messages %}",
            "{{- message.role }}={{ message.content }}",
            "{%- if not loop.last %};{% endif %}",
            "{%- endfor %}",
            "{%- if add_generation_prompt %}|assistant{% endif %}"
        );
        let output = render(
            template,
            &[ChatMessage::system("be terse"), ChatMessage::user("hello")],
            true,
        );
        assert_eq!(output, "system=be terse;user=hello|assistant");
    }

    #[test]
    fn trims_jinja_whitespace_markers() {
        let template = "a\n  {%- if true -%}\n  b\n  {%- endif -%}\n  c";
        assert_eq!(render(template, &[], false), "abc");
    }

    #[test]
    fn supports_indexing_conditions_and_defined_test() {
        let template = concat!(
            "{%- if messages[0]['role'] == 'system' %}",
            "{{ messages[0].content }}",
            "{%- else %}default{% endif %}",
            "{%- if messages[0].missing is defined %}bad{% endif %}"
        );
        assert_eq!(
            render(template, &[ChatMessage::system("system text")], false),
            "system text"
        );
        assert_eq!(
            render(template, &[ChatMessage::user("user text")], false),
            "default"
        );
    }

    #[test]
    fn supports_set_and_tojson() {
        let template = concat!(
            "{%- set item = messages[0] %}",
            "{{- item.role }}:{{ item | tojson }}"
        );
        assert_eq!(
            render(template, &[ChatMessage::user("hi \"there\"")], false),
            "user:{\"content\":\"hi \\\"there\\\"\",\"role\":\"user\"}"
        );
    }

    #[test]
    fn unsupported_construct_fails_explicitly() {
        let error = ChatTemplate::parse("{% macro thing() %}x{% endmacro %}").unwrap_err();
        assert!(matches!(error, ChatTemplateError::Unsupported(_)));
    }

    #[test]
    fn qwen_style_template_selects_default_system_prompt() {
        let template = concat!(
            "{%- if messages[0]['role'] == 'system' %}",
            "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}",
            "{%- else %}",
            "{{- '<|im_start|>system\\nYou are helpful.<|im_end|>\\n' }}",
            "{%- endif %}",
            "{%- for message in messages %}",
            "{%- if (message.role == 'user') or (message.role == 'system' and not loop.first) %}",
            "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}",
            "{%- endif %}",
            "{%- endfor %}",
            "{%- if add_generation_prompt %}",
            "{{- '<|im_start|>assistant\\n' }}",
            "{%- endif %}"
        );
        assert_eq!(
            render(template, &[ChatMessage::user("Hello")], true),
            "<|im_start|>system\nYou are helpful.<|im_end|>\n\
             <|im_start|>user\nHello<|im_end|>\n\
             <|im_start|>assistant\n"
        );
        assert_eq!(
            render(
                template,
                &[ChatMessage::system("Custom"), ChatMessage::user("Hello")],
                false
            ),
            "<|im_start|>system\nCustom<|im_end|>\n\
             <|im_start|>user\nHello<|im_end|>\n"
        );
    }
}
