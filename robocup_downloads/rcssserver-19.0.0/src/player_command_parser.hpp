/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

#ifndef YY_YY_PLAYER_COMMAND_PARSER_HPP_INCLUDED
# define YY_YY_PLAYER_COMMAND_PARSER_HPP_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token kinds.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    YYEMPTY = -2,
    YYEOF = 0,                     /* "end of file"  */
    YYerror = 256,                 /* error  */
    YYUNDEF = 257,                 /* "invalid token"  */
    RCSS_PCOM_INT = 258,           /* RCSS_PCOM_INT  */
    RCSS_PCOM_REAL = 259,          /* RCSS_PCOM_REAL  */
    RCSS_PCOM_STR = 260,           /* RCSS_PCOM_STR  */
    RCSS_PCOM_LP = 261,            /* "("  */
    RCSS_PCOM_RP = 262,            /* ")"  */
    RCSS_PCOM_DASH = 263,          /* "dash"  */
    RCSS_PCOM_TURN = 264,          /* "turn"  */
    RCSS_PCOM_TURN_NECK = 265,     /* "turn_neck"  */
    RCSS_PCOM_CHANGE_FOCUS = 266,  /* "change_focus"  */
    RCSS_PCOM_KICK = 267,          /* "kick"  */
    RCSS_PCOM_LONG_KICK = 268,     /* "long_kick"  */
    RCSS_PCOM_CATCH = 269,         /* "catch"  */
    RCSS_PCOM_DROP = 270,          /* "drop"  */
    RCSS_PCOM_SAY = 271,           /* "say"  */
    RCSS_PCOM_UNQ_SAY = 272,       /* "unquoted say"  */
    RCSS_PCOM_SENSE_BODY = 273,    /* "sense_body"  */
    RCSS_PCOM_SCORE = 274,         /* "score"  */
    RCSS_PCOM_MOVE = 275,          /* "move"  */
    RCSS_PCOM_CHANGE_VIEW = 276,   /* "change_view"  */
    RCSS_PCOM_COMPRESSION = 277,   /* "compression"  */
    RCSS_PCOM_BYE = 278,           /* "bye"  */
    RCSS_PCOM_DONE = 279,          /* "done"  */
    RCSS_PCOM_POINTTO = 280,       /* "pointto"  */
    RCSS_PCOM_ATTENTIONTO = 281,   /* "attentionto"  */
    RCSS_PCOM_TACKLE = 282,        /* "tackle"  */
    RCSS_PCOM_CLANG = 283,         /* "clang"  */
    RCSS_PCOM_EAR = 284,           /* "ear"  */
    RCSS_PCOM_SYNCH_SEE = 285,     /* "synch_see"  */
    RCSS_PCOM_GAUSSIAN_SEE = 286,  /* "gaussian_see"  */
    RCSS_PCOM_VIEW_WIDTH_NARROW = 287, /* "narrow"  */
    RCSS_PCOM_VIEW_WIDTH_NORMAL = 288, /* "normal"  */
    RCSS_PCOM_VIEW_WIDTH_WIDE = 289, /* "wide"  */
    RCSS_PCOM_VIEW_QUALITY_LOW = 290, /* "low"  */
    RCSS_PCOM_VIEW_QUALITY_HIGH = 291, /* "high"  */
    RCSS_PCOM_ON = 292,            /* "on"  */
    RCSS_PCOM_OFF = 293,           /* "off"  */
    RCSS_PCOM_TRUE = 294,          /* "true"  */
    RCSS_PCOM_FALSE = 295,         /* "false"  */
    RCSS_PCOM_OUR = 296,           /* "our"  */
    RCSS_PCOM_OPP = 297,           /* "opp"  */
    RCSS_PCOM_LEFT = 298,          /* RCSS_PCOM_LEFT  */
    RCSS_PCOM_RIGHT = 299,         /* RCSS_PCOM_RIGHT  */
    RCSS_PCOM_EAR_PARTIAL = 300,   /* "partial"  */
    RCSS_PCOM_EAR_COMPLETE = 301,  /* "complete"  */
    RCSS_PCOM_CLANG_VERSION = 302, /* "ver"  */
    RCSS_PCOM_ERROR = 303          /* RCSS_PCOM_ERROR  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */




int yyparse (rcss::pcom::Parser::Param& param);


#endif /* !YY_YY_PLAYER_COMMAND_PARSER_HPP_INCLUDED  */
