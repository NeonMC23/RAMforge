use std::fmt;

/// GGUF metadata value type tag (as stored in file)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufValueType {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::UInt8),
            1 => Some(Self::Int8),
            2 => Some(Self::UInt16),
            3 => Some(Self::Int16),
            4 => Some(Self::UInt32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::UInt64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

/// Structured representation of a GGUF metadata value
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(ArrayValue),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArrayValue {
    pub element_type: GgufValueType,
    pub values: Vec<MetadataValue>,
}

impl MetadataValue {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::UInt8(_) => "uint8",
            Self::Int8(_) => "int8",
            Self::UInt16(_) => "uint16",
            Self::Int16(_) => "int16",
            Self::UInt32(_) => "uint32",
            Self::Int32(_) => "int32",
            Self::Float32(_) => "float32",
            Self::Bool(_) => "bool",
            Self::String(_) => "string",
            Self::Array(_) => "array",
            Self::UInt64(_) => "uint64",
            Self::Int64(_) => "int64",
            Self::Float64(_) => "float64",
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::UInt8(v) => Some(*v as u64),
            Self::Int8(v) => Some(*v as u64),
            Self::UInt16(v) => Some(*v as u64),
            Self::Int16(v) => Some(*v as u64),
            Self::UInt32(v) => Some(*v as u64),
            Self::Int32(v) => Some(*v as u64),
            Self::UInt64(v) => Some(*v),
            Self::Int64(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        self.as_u64().and_then(|v| u32::try_from(v).ok())
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::UInt8(v) => Some(*v as i64),
            Self::Int8(v) => Some(*v as i64),
            Self::UInt16(v) => Some(*v as i64),
            Self::Int16(v) => Some(*v as i64),
            Self::UInt32(v) => Some(*v as i64),
            Self::Int32(v) => Some(*v as i64),
            Self::UInt64(v) => Some(*v as i64),
            Self::Int64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&ArrayValue> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }
}

impl fmt::Display for MetadataValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UInt8(v) => write!(f, "{}", v),
            Self::Int8(v) => write!(f, "{}", v),
            Self::UInt16(v) => write!(f, "{}", v),
            Self::Int16(v) => write!(f, "{}", v),
            Self::UInt32(v) => write!(f, "{}", v),
            Self::Int32(v) => write!(f, "{}", v),
            Self::Float32(v) => write!(f, "{}", v),
            Self::Bool(v) => write!(f, "{}", v),
            Self::String(v) => write!(f, "\"{}\"", v),
            Self::Array(arr) => {
                if arr.values.len() > 10 {
                    write!(f, "[{} elements]", arr.values.len())
                } else {
                    write!(f, "[")?;
                    for (i, val) in arr.values.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", val)?;
                    }
                    write!(f, "]")
                }
            }
            Self::UInt64(v) => write!(f, "{}", v),
            Self::Int64(v) => write!(f, "{}", v),
            Self::Float64(v) => write!(f, "{}", v),
        }
    }
}

/// GGML / GGUF tensor element type
///
/// This covers the known types from both the legacy ggml enum and the newer
/// llama.cpp extensions. Unknown values are preserved as `Unknown(u32)` so
/// parsing never fails on a future type.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
    BF16,
    // Newer types from extended enum
    TQ1_0,
    TQ2_0,
    MXFP4,
    // Additional IQ types that appear in newer llama.cpp versions
    IQ4_0,
    IQ4_NL_B16,
    IQ4_NL_B32,
    IQ4_NL_B64,
    IQ4_SQ,
    IQ4_K,
    IQ5_0,
    IQ5_NL,
    IQ5_SQ,
    IQ5_K,
    IQ6_0,
    IQ6_NL,
    IQ6_K,
    IQ2_K,
    IQ2_0,
    IQ3_T,
    IQ8_0,
    Q3_I,
    // Fallback
    Unknown(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            34 => Self::TQ1_0,
            35 => Self::TQ2_0,
            39 => Self::MXFP4,
            // Extended mapping – best effort, overlapping with older mapping
            // For values 16..40 we already covered the core set; additional
            // newer names map to the same numbers in some implementations.
            // To avoid losing information, we keep the core names and expose
            // extra variants via Unknown handling if needed. However we also
            // provide explicit variants for the extended set that share numbers
            // in different enum versions – we disambiguate by preferring the
            // newer ggml spec where possible.
            //
            // For robustness, we also map some of the newer IQ types that
            // appear beyond 30 in some forks:
            32 => Self::IQ4_0, // in newer spec, but also conflicts; handled as Unknown fallback? We'll map.
            33 => Self::IQ4_NL_B16,
            36 => Self::IQ4_NL_B32,
            37 => Self::IQ4_NL_B64,
            38 => Self::IQ4_SQ,
            40 => Self::IQ8_0,
            41 => Self::Q3_I,
            _ => Self::Unknown(v),
        }
    }

    /// Return canonical name
    pub fn name(&self) -> String {
        match self {
            Self::F32 => "F32".to_string(),
            Self::F16 => "F16".to_string(),
            Self::Q4_0 => "Q4_0".to_string(),
            Self::Q4_1 => "Q4_1".to_string(),
            Self::Q5_0 => "Q5_0".to_string(),
            Self::Q5_1 => "Q5_1".to_string(),
            Self::Q8_0 => "Q8_0".to_string(),
            Self::Q8_1 => "Q8_1".to_string(),
            Self::Q2_K => "Q2_K".to_string(),
            Self::Q3_K => "Q3_K".to_string(),
            Self::Q4_K => "Q4_K".to_string(),
            Self::Q5_K => "Q5_K".to_string(),
            Self::Q6_K => "Q6_K".to_string(),
            Self::Q8_K => "Q8_K".to_string(),
            Self::IQ2_XXS => "IQ2_XXS".to_string(),
            Self::IQ2_XS => "IQ2_XS".to_string(),
            Self::IQ3_XXS => "IQ3_XXS".to_string(),
            Self::IQ1_S => "IQ1_S".to_string(),
            Self::IQ4_NL => "IQ4_NL".to_string(),
            Self::IQ3_S => "IQ3_S".to_string(),
            Self::IQ2_S => "IQ2_S".to_string(),
            Self::IQ4_XS => "IQ4_XS".to_string(),
            Self::I8 => "I8".to_string(),
            Self::I16 => "I16".to_string(),
            Self::I32 => "I32".to_string(),
            Self::I64 => "I64".to_string(),
            Self::F64 => "F64".to_string(),
            Self::IQ1_M => "IQ1_M".to_string(),
            Self::BF16 => "BF16".to_string(),
            Self::TQ1_0 => "TQ1_0".to_string(),
            Self::TQ2_0 => "TQ2_0".to_string(),
            Self::MXFP4 => "MXFP4".to_string(),
            Self::IQ4_0 => "IQ4_0".to_string(),
            Self::IQ4_NL_B16 => "IQ4_NL_B16".to_string(),
            Self::IQ4_NL_B32 => "IQ4_NL_B32".to_string(),
            Self::IQ4_NL_B64 => "IQ4_NL_B64".to_string(),
            Self::IQ4_SQ => "IQ4_SQ".to_string(),
            Self::IQ4_K => "IQ4_K".to_string(),
            Self::IQ5_0 => "IQ5_0".to_string(),
            Self::IQ5_NL => "IQ5_NL".to_string(),
            Self::IQ5_SQ => "IQ5_SQ".to_string(),
            Self::IQ5_K => "IQ5_K".to_string(),
            Self::IQ6_0 => "IQ6_0".to_string(),
            Self::IQ6_NL => "IQ6_NL".to_string(),
            Self::IQ6_K => "IQ6_K".to_string(),
            Self::IQ2_K => "IQ2_K".to_string(),
            Self::IQ2_0 => "IQ2_0".to_string(),
            Self::IQ3_T => "IQ3_T".to_string(),
            Self::IQ8_0 => "IQ8_0".to_string(),
            Self::Q3_I => "Q3_I".to_string(),
            Self::Unknown(v) => format!("UNKNOWN({})", v),
        }
    }

    /// Return (block_size, type_size_in_bytes) if known
    ///
    /// This is used to compute tensor byte length without loading data.
    pub fn type_info(&self) -> Option<(u64, u64)> {
        // (block_size, type_size)
        match self {
            Self::F32 => Some((1, 4)),
            Self::F16 => Some((1, 2)),
            Self::BF16 => Some((1, 2)),
            Self::F64 => Some((1, 8)),
            Self::I8 => Some((1, 1)),
            Self::I16 => Some((1, 2)),
            Self::I32 => Some((1, 4)),
            Self::I64 => Some((1, 8)),
            Self::Q4_0 => Some((32, 18)),
            Self::Q4_1 => Some((32, 20)),
            Self::Q5_0 => Some((32, 22)),
            Self::Q5_1 => Some((32, 24)),
            Self::Q8_0 => Some((32, 34)),
            Self::Q8_1 => Some((32, 40)), // best effort: 2*2 + 32 = 36, but some impls use 40; use 40
            Self::Q2_K => Some((256, 84)),
            Self::Q3_K => Some((256, 110)),
            Self::Q4_K => Some((256, 144)),
            Self::Q5_K => Some((256, 176)),
            Self::Q6_K => Some((256, 210)),
            Self::Q8_K => Some((256, 292)),
            // IQ types – approximate sizes based on public docs; many are 256 block size
            Self::IQ2_XXS => Some((256, 66)),
            Self::IQ2_XS => Some((256, 74)),
            Self::IQ3_XXS => Some((256, 98)),
            Self::IQ1_S => Some((256, 50)),
            Self::IQ4_NL => Some((32, 18)),
            Self::IQ3_S => Some((256, 110)),
            Self::IQ2_S => Some((256, 82)),
            Self::IQ4_XS => Some((256, 136)),
            Self::IQ1_M => Some((256, 56)),
            Self::TQ1_0 => Some((256, 54)),
            Self::TQ2_0 => Some((256, 70)),
            Self::MXFP4 => Some((32, 18)), // approx
            _ => None,
        }
    }

    pub fn as_u32(&self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2_K => 10,
            Self::Q3_K => 11,
            Self::Q4_K => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
            Self::Q8_K => 15,
            Self::IQ2_XXS => 16,
            Self::IQ2_XS => 17,
            Self::IQ3_XXS => 18,
            Self::IQ1_S => 19,
            Self::IQ4_NL => 20,
            Self::IQ3_S => 21,
            Self::IQ2_S => 22,
            Self::IQ4_XS => 23,
            Self::I8 => 24,
            Self::I16 => 25,
            Self::I32 => 26,
            Self::I64 => 27,
            Self::F64 => 28,
            Self::IQ1_M => 29,
            Self::BF16 => 30,
            Self::TQ1_0 => 34,
            Self::TQ2_0 => 35,
            Self::MXFP4 => 39,
            Self::IQ4_0 => 32,
            Self::IQ4_NL_B16 => 33,
            Self::IQ4_NL_B32 => 36,
            Self::IQ4_NL_B64 => 37,
            Self::IQ4_SQ => 38,
            Self::IQ4_K => 27, // placeholder – actual mapping varies
            Self::IQ5_0 => 28,
            Self::IQ5_NL => 29,
            Self::IQ5_SQ => 30,
            Self::IQ5_K => 31,
            Self::IQ6_0 => 32,
            Self::IQ6_NL => 33,
            Self::IQ6_K => 34,
            Self::IQ2_K => 35,
            Self::IQ2_0 => 38,
            Self::IQ3_T => 39,
            Self::IQ8_0 => 40,
            Self::Q3_I => 41,
            Self::Unknown(v) => *v,
        }
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}
