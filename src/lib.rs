#![expect(clippy::type_complexity)] // Necessary for mapping expressions.
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

mod codegen;
