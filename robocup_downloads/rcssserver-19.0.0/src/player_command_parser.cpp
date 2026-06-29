/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison implementation for Yacc-like parsers in C

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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output, and Bison version.  */
#define YYBISON 30802

/* Bison version string.  */
#define YYBISON_VERSION "3.8.2"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 1

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* First part of user prologue.  */
#line 26 "player_command_parser.ypp"

#include "pcombuilder.h"
#include "pcomparser.h"


#define    yyparse    RCSS_PCOM_parse

void yyerror( rcss::pcom::Parser::Param& param, const char* s );
int yyerror( rcss::pcom::Parser::Param& param, char* s );

namespace
{
  inline rcss::pcom::Builder& getBuilder( rcss::pcom::Parser::Param& param )
  {
    return param.getBuilder();
  }

#define YYSTYPE rcss::pcom::Parser::Lexer::Holder

  inline int yylex( YYSTYPE* holder, rcss::pcom::Parser::Param& param )
  {
    int rval = param.getLexer().lex( *holder );
//    cout << rval << endl;
    return rval;
  }

}

#define BUILDER getBuilder( param )


#line 103 "player_command_parser.cpp"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

/* Use api.header.include to #include this header
   instead of duplicating it here.  */
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
/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
  YYSYMBOL_YYerror = 1,                    /* error  */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
  YYSYMBOL_RCSS_PCOM_INT = 3,              /* RCSS_PCOM_INT  */
  YYSYMBOL_RCSS_PCOM_REAL = 4,             /* RCSS_PCOM_REAL  */
  YYSYMBOL_RCSS_PCOM_STR = 5,              /* RCSS_PCOM_STR  */
  YYSYMBOL_RCSS_PCOM_LP = 6,               /* "("  */
  YYSYMBOL_RCSS_PCOM_RP = 7,               /* ")"  */
  YYSYMBOL_RCSS_PCOM_DASH = 8,             /* "dash"  */
  YYSYMBOL_RCSS_PCOM_TURN = 9,             /* "turn"  */
  YYSYMBOL_RCSS_PCOM_TURN_NECK = 10,       /* "turn_neck"  */
  YYSYMBOL_RCSS_PCOM_CHANGE_FOCUS = 11,    /* "change_focus"  */
  YYSYMBOL_RCSS_PCOM_KICK = 12,            /* "kick"  */
  YYSYMBOL_RCSS_PCOM_LONG_KICK = 13,       /* "long_kick"  */
  YYSYMBOL_RCSS_PCOM_CATCH = 14,           /* "catch"  */
  YYSYMBOL_RCSS_PCOM_DROP = 15,            /* "drop"  */
  YYSYMBOL_RCSS_PCOM_SAY = 16,             /* "say"  */
  YYSYMBOL_RCSS_PCOM_UNQ_SAY = 17,         /* "unquoted say"  */
  YYSYMBOL_RCSS_PCOM_SENSE_BODY = 18,      /* "sense_body"  */
  YYSYMBOL_RCSS_PCOM_SCORE = 19,           /* "score"  */
  YYSYMBOL_RCSS_PCOM_MOVE = 20,            /* "move"  */
  YYSYMBOL_RCSS_PCOM_CHANGE_VIEW = 21,     /* "change_view"  */
  YYSYMBOL_RCSS_PCOM_COMPRESSION = 22,     /* "compression"  */
  YYSYMBOL_RCSS_PCOM_BYE = 23,             /* "bye"  */
  YYSYMBOL_RCSS_PCOM_DONE = 24,            /* "done"  */
  YYSYMBOL_RCSS_PCOM_POINTTO = 25,         /* "pointto"  */
  YYSYMBOL_RCSS_PCOM_ATTENTIONTO = 26,     /* "attentionto"  */
  YYSYMBOL_RCSS_PCOM_TACKLE = 27,          /* "tackle"  */
  YYSYMBOL_RCSS_PCOM_CLANG = 28,           /* "clang"  */
  YYSYMBOL_RCSS_PCOM_EAR = 29,             /* "ear"  */
  YYSYMBOL_RCSS_PCOM_SYNCH_SEE = 30,       /* "synch_see"  */
  YYSYMBOL_RCSS_PCOM_GAUSSIAN_SEE = 31,    /* "gaussian_see"  */
  YYSYMBOL_RCSS_PCOM_VIEW_WIDTH_NARROW = 32, /* "narrow"  */
  YYSYMBOL_RCSS_PCOM_VIEW_WIDTH_NORMAL = 33, /* "normal"  */
  YYSYMBOL_RCSS_PCOM_VIEW_WIDTH_WIDE = 34, /* "wide"  */
  YYSYMBOL_RCSS_PCOM_VIEW_QUALITY_LOW = 35, /* "low"  */
  YYSYMBOL_RCSS_PCOM_VIEW_QUALITY_HIGH = 36, /* "high"  */
  YYSYMBOL_RCSS_PCOM_ON = 37,              /* "on"  */
  YYSYMBOL_RCSS_PCOM_OFF = 38,             /* "off"  */
  YYSYMBOL_RCSS_PCOM_TRUE = 39,            /* "true"  */
  YYSYMBOL_RCSS_PCOM_FALSE = 40,           /* "false"  */
  YYSYMBOL_RCSS_PCOM_OUR = 41,             /* "our"  */
  YYSYMBOL_RCSS_PCOM_OPP = 42,             /* "opp"  */
  YYSYMBOL_RCSS_PCOM_LEFT = 43,            /* RCSS_PCOM_LEFT  */
  YYSYMBOL_44_l_ = 44,                     /* 'l'  */
  YYSYMBOL_RCSS_PCOM_RIGHT = 45,           /* RCSS_PCOM_RIGHT  */
  YYSYMBOL_46_r_ = 46,                     /* 'r'  */
  YYSYMBOL_RCSS_PCOM_EAR_PARTIAL = 47,     /* "partial"  */
  YYSYMBOL_RCSS_PCOM_EAR_COMPLETE = 48,    /* "complete"  */
  YYSYMBOL_RCSS_PCOM_CLANG_VERSION = 49,   /* "ver"  */
  YYSYMBOL_RCSS_PCOM_ERROR = 50,           /* RCSS_PCOM_ERROR  */
  YYSYMBOL_YYACCEPT = 51,                  /* $accept  */
  YYSYMBOL_command_list = 52,              /* command_list  */
  YYSYMBOL_command = 53,                   /* command  */
  YYSYMBOL_dash_com = 54,                  /* dash_com  */
  YYSYMBOL_dash_params = 55,               /* dash_params  */
  YYSYMBOL_dash_left = 56,                 /* dash_left  */
  YYSYMBOL_dash_right = 57,                /* dash_right  */
  YYSYMBOL_turn_com = 58,                  /* turn_com  */
  YYSYMBOL_turn_neck_com = 59,             /* turn_neck_com  */
  YYSYMBOL_change_focus_com = 60,          /* change_focus_com  */
  YYSYMBOL_kick_com = 61,                  /* kick_com  */
  YYSYMBOL_long_kick_com = 62,             /* long_kick_com  */
  YYSYMBOL_catch_com = 63,                 /* catch_com  */
  YYSYMBOL_drop_com = 64,                  /* drop_com  */
  YYSYMBOL_say_com = 65,                   /* say_com  */
  YYSYMBOL_sense_body_com = 66,            /* sense_body_com  */
  YYSYMBOL_score_com = 67,                 /* score_com  */
  YYSYMBOL_move_com = 68,                  /* move_com  */
  YYSYMBOL_change_view_com = 69,           /* change_view_com  */
  YYSYMBOL_view_width = 70,                /* view_width  */
  YYSYMBOL_view_quality = 71,              /* view_quality  */
  YYSYMBOL_compression_com = 72,           /* compression_com  */
  YYSYMBOL_bye_com = 73,                   /* bye_com  */
  YYSYMBOL_done_com = 74,                  /* done_com  */
  YYSYMBOL_pointto_com = 75,               /* pointto_com  */
  YYSYMBOL_attentionto_com = 76,           /* attentionto_com  */
  YYSYMBOL_tackle_com = 77,                /* tackle_com  */
  YYSYMBOL_clang_com = 78,                 /* clang_com  */
  YYSYMBOL_ear_com = 79,                   /* ear_com  */
  YYSYMBOL_synch_see_com = 80,             /* synch_see_com  */
  YYSYMBOL_gaussian_see_com = 81,          /* gaussian_see_com  */
  YYSYMBOL_on_off = 82,                    /* on_off  */
  YYSYMBOL_boolean_value = 83,             /* boolean_value  */
  YYSYMBOL_team_side = 84,                 /* team_side  */
  YYSYMBOL_partial_complete = 85,          /* partial_complete  */
  YYSYMBOL_floating_point_number = 86      /* floating_point_number  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;




#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

/* Work around bug in HP-UX 11.23, which defines these macros
   incorrectly for preprocessor constants.  This workaround can likely
   be removed in 2023, as HPE has promised support for HP-UX 11.23
   (aka HP-UX 11i v2) only through the end of 2022; see Table 2 of
   <https://h20195.www2.hpe.com/V2/getpdf.aspx/4AA4-7673ENW.pdf>.  */
#ifdef __hpux
# undef UINT_LEAST8_MAX
# undef UINT_LEAST16_MAX
# define UINT_LEAST8_MAX 255
# define UINT_LEAST16_MAX 65535
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_uint8 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YY_USE(E) ((void) (E))
#else
# define YY_USE(E) /* empty */
#endif

/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
#if defined __GNUC__ && ! defined __ICC && 406 <= __GNUC__ * 100 + __GNUC_MINOR__
# if __GNUC__ * 100 + __GNUC_MINOR__ < 407
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")
# else
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# endif
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if !defined yyoverflow

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* !defined yyoverflow */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  51
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   149

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  51
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  36
/* YYNRULES -- Number of rules.  */
#define YYNRULES  87
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  168

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   303


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : YYSYMBOL_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    44,     2,
       2,     2,     2,     2,    46,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    45,
      47,    48,    49,    50
};

#if YYDEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,   131,   131,   132,   135,   136,   137,   138,   139,   140,
     141,   142,   143,   144,   145,   146,   147,   148,   149,   150,
     151,   152,   153,   154,   155,   156,   157,   160,   164,   168,
     171,   172,   173,   174,   177,   183,   189,   195,   201,   207,
     213,   219,   225,   231,   235,   241,   247,   253,   259,   263,
     269,   273,   277,   283,   287,   293,   299,   305,   311,   315,
     321,   325,   329,   335,   340,   346,   352,   356,   360,   364,
     368,   372,   378,   384,   390,   394,   400,   404,   408,   412,
     418,   422,   426,   430,   436,   440,   446,   450
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if YYDEBUG || 0
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "\"end of file\"", "error", "\"invalid token\"", "RCSS_PCOM_INT",
  "RCSS_PCOM_REAL", "RCSS_PCOM_STR", "\"(\"", "\")\"", "\"dash\"",
  "\"turn\"", "\"turn_neck\"", "\"change_focus\"", "\"kick\"",
  "\"long_kick\"", "\"catch\"", "\"drop\"", "\"say\"", "\"unquoted say\"",
  "\"sense_body\"", "\"score\"", "\"move\"", "\"change_view\"",
  "\"compression\"", "\"bye\"", "\"done\"", "\"pointto\"",
  "\"attentionto\"", "\"tackle\"", "\"clang\"", "\"ear\"", "\"synch_see\"",
  "\"gaussian_see\"", "\"narrow\"", "\"normal\"", "\"wide\"", "\"low\"",
  "\"high\"", "\"on\"", "\"off\"", "\"true\"", "\"false\"", "\"our\"",
  "\"opp\"", "RCSS_PCOM_LEFT", "'l'", "RCSS_PCOM_RIGHT", "'r'",
  "\"partial\"", "\"complete\"", "\"ver\"", "RCSS_PCOM_ERROR", "$accept",
  "command_list", "command", "dash_com", "dash_params", "dash_left",
  "dash_right", "turn_com", "turn_neck_com", "change_focus_com",
  "kick_com", "long_kick_com", "catch_com", "drop_com", "say_com",
  "sense_body_com", "score_com", "move_com", "change_view_com",
  "view_width", "view_quality", "compression_com", "bye_com", "done_com",
  "pointto_com", "attentionto_com", "tackle_com", "clang_com", "ear_com",
  "synch_see_com", "gaussian_see_com", "on_off", "boolean_value",
  "team_side", "partial_complete", "floating_point_number", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#define YYPACT_NINF (-124)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-1)

#define yytable_value_is_error(Yyn) \
  0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int16 yypact[] =
{
       3,    57,  -124,    12,  -124,  -124,  -124,  -124,  -124,  -124,
    -124,  -124,  -124,  -124,  -124,  -124,  -124,  -124,  -124,  -124,
    -124,  -124,  -124,  -124,  -124,  -124,  -124,  -124,    88,    28,
      28,    28,    28,    28,    28,     7,    11,    14,    29,    28,
      63,    71,    91,    92,    23,    17,    28,    96,    97,    98,
      99,  -124,  -124,  -124,  -124,     2,   100,   102,   103,    86,
     104,   105,    28,    28,    28,   106,  -124,   107,  -124,  -124,
      28,  -124,  -124,  -124,    21,   108,  -124,  -124,   109,    28,
     101,   110,  -124,  -124,  -124,  -124,   115,     0,    61,   -13,
    -124,  -124,    28,    28,  -124,    74,  -124,    77,  -124,  -124,
     114,  -124,  -124,   116,   117,   118,  -124,  -124,   119,  -124,
    -124,  -124,   120,  -124,  -124,   121,   122,  -124,   123,  -124,
    -124,  -124,  -124,  -124,   124,   129,  -124,  -124,     1,    28,
      28,  -124,  -124,  -124,  -124,  -124,  -124,  -124,  -124,  -124,
    -124,   130,     4,   127,  -124,  -124,     6,   128,   131,   132,
     133,   134,   135,  -124,   136,   137,   138,  -124,  -124,   139,
    -124,   140,  -124,   141,  -124,  -124,  -124,  -124
};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_int8 yydefact[] =
{
       0,     0,    43,     0,     2,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     1,     3,    86,    87,     0,     0,    30,    31,     0,
       0,     0,     0,     0,     0,     0,    42,     0,    45,    46,
       0,    50,    51,    52,     0,     0,    56,    57,     0,     0,
       0,     0,    80,    81,    82,    83,     0,     0,     0,     0,
      72,    73,     0,     0,    29,     0,    32,     0,    33,    27,
       0,    36,    37,     0,     0,     0,    41,    44,     0,    49,
      53,    54,     0,    55,    59,     0,     0,    62,     0,    63,
      76,    77,    78,    79,     0,     0,    74,    75,     0,     0,
       0,    28,    38,    39,    40,    47,    48,    58,    61,    60,
      64,     0,     0,     0,    84,    85,     0,     0,     0,     0,
       0,     0,     0,    71,     0,     0,     0,    34,    35,     0,
      69,     0,    68,     0,    70,    65,    67,    66
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -124,  -124,   146,  -124,  -124,    64,    79,  -124,  -124,  -124,
    -124,  -124,  -124,  -124,  -124,  -124,  -124,  -124,  -124,  -124,
    -124,  -124,  -124,  -124,  -124,  -124,  -124,  -124,  -124,  -124,
    -124,  -124,  -124,     9,  -123,   -29
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_uint8 yydefgoto[] =
{
       0,     3,     4,     5,    56,    57,    58,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    74,
     112,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,   128,   124,    86,   147,    59
};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint8 yytable[] =
{
      60,    61,    62,    63,    64,    65,   142,   119,   143,     1,
      70,   151,    51,   154,    66,    79,    67,    87,     1,   152,
       2,    68,    80,   155,   126,   127,    53,    54,   109,     2,
     100,    53,    54,   103,   104,   105,    69,   120,   121,   122,
     123,   108,    82,    83,    84,    92,    85,    93,   144,   145,
     115,   144,   145,   144,   145,    81,   110,   111,    82,    83,
      84,    78,    85,   129,   130,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    75,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    53,
      54,    53,    54,    99,    55,    71,    72,    73,    76,    77,
     148,   149,    88,    89,   116,    90,    91,    94,    95,    97,
     125,   101,   102,   106,   107,   113,   114,   117,   118,    93,
      92,   131,    98,   132,   133,   134,   135,   136,   137,   138,
     139,   140,   141,   150,   153,   156,    96,   146,   157,   158,
     159,   160,   161,   162,   163,   164,   165,   166,   167,    52
};

static const yytype_uint8 yycheck[] =
{
      29,    30,    31,    32,    33,    34,     5,     7,     7,     6,
      39,     7,     0,     7,     7,    44,     5,    46,     6,   142,
      17,     7,     5,   146,    37,    38,     3,     4,     7,    17,
      59,     3,     4,    62,    63,    64,     7,    37,    38,    39,
      40,    70,    41,    42,    43,    43,    45,    45,    47,    48,
      79,    47,    48,    47,    48,    38,    35,    36,    41,    42,
      43,    38,    45,    92,    93,     8,     9,    10,    11,    12,
      13,    14,    15,    16,     3,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,     3,
       4,     3,     4,     7,     6,    32,    33,    34,     7,     7,
     129,   130,     6,     6,     3,     7,     7,     7,     6,     6,
      49,     7,     7,     7,     7,     7,     7,     7,     3,    45,
      43,     7,    58,     7,     7,     7,     7,     7,     7,     7,
       7,     7,     3,     3,     7,     7,    57,   128,     7,     7,
       7,     7,     7,     7,     7,     7,     7,     7,     7,     3
};

/* YYSTOS[STATE-NUM] -- The symbol kind of the accessing symbol of
   state STATE-NUM.  */
static const yytype_int8 yystos[] =
{
       0,     6,    17,    52,    53,    54,    58,    59,    60,    61,
      62,    63,    64,    65,    66,    67,    68,    69,    72,    73,
      74,    75,    76,    77,    78,    79,    80,    81,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,     0,    53,     3,     4,     6,    55,    56,    57,    86,
      86,    86,    86,    86,    86,    86,     7,     5,     7,     7,
      86,    32,    33,    34,    70,     3,     7,     7,    38,    86,
       5,    38,    41,    42,    43,    45,    84,    86,     6,     6,
       7,     7,    43,    45,     7,     6,    57,     6,    56,     7,
      86,     7,     7,    86,    86,    86,     7,     7,    86,     7,
      35,    36,    71,     7,     7,    86,     3,     7,     3,     7,
      37,    38,    39,    40,    83,    49,    37,    38,    82,    86,
      86,     7,     7,     7,     7,     7,     7,     7,     7,     7,
       7,     3,     5,     7,    47,    48,    84,    85,    86,    86,
       3,     7,    85,     7,     7,    85,     7,     7,     7,     7,
       7,     7,     7,     7,     7,     7,     7,     7
};

/* YYR1[RULE-NUM] -- Symbol kind of the left-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr1[] =
{
       0,    51,    52,    52,    53,    53,    53,    53,    53,    53,
      53,    53,    53,    53,    53,    53,    53,    53,    53,    53,
      53,    53,    53,    53,    53,    53,    53,    54,    54,    54,
      55,    55,    55,    55,    56,    57,    58,    59,    60,    61,
      62,    63,    64,    65,    65,    66,    67,    68,    69,    69,
      70,    70,    70,    71,    71,    72,    73,    74,    75,    75,
      76,    76,    76,    77,    77,    78,    79,    79,    79,    79,
      79,    79,    80,    81,    82,    82,    83,    83,    83,    83,
      84,    84,    84,    84,    85,    85,    86,    86
};

/* YYR2[RULE-NUM] -- Number of symbols on the right-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     1,     2,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     4,     5,     4,
       1,     1,     2,     2,     5,     5,     4,     4,     5,     5,
       5,     4,     3,     1,     4,     3,     3,     5,     5,     4,
       1,     1,     1,     1,     1,     4,     3,     3,     5,     4,
       5,     5,     4,     4,     5,     8,     8,     8,     7,     7,
       7,     6,     3,     3,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYNOMEM         goto yyexhaustedlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (param, YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use YYerror or YYUNDEF. */
#define YYERRCODE YYUNDEF


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)




# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value, param); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, rcss::pcom::Parser::Param& param)
{
  FILE *yyoutput = yyo;
  YY_USE (yyoutput);
  YY_USE (param);
  if (!yyvaluep)
    return;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, rcss::pcom::Parser::Param& param)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  yy_symbol_value_print (yyo, yykind, yyvaluep, param);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp,
                 int yyrule, rcss::pcom::Parser::Param& param)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)], param);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule, param); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif






/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep, rcss::pcom::Parser::Param& param)
{
  YY_USE (yyvaluep);
  YY_USE (param);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}






/*----------.
| yyparse.  |
`----------*/

int
yyparse (rcss::pcom::Parser::Param& param)
{
/* Lookahead token kind.  */
int yychar;


/* The semantic value of the lookahead symbol.  */
/* Default value used for initialization, for pacifying older GCCs
   or non-GCC compilers.  */
YY_INITIAL_VALUE (static YYSTYPE yyval_default;)
YYSTYPE yylval YY_INITIAL_VALUE (= yyval_default);

    /* Number of syntax errors so far.  */
    int yynerrs = 0;

    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead symbol kind.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = YYEMPTY; /* Cause a token to be read.  */

  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    YYNOMEM;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        YYNOMEM;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          YYNOMEM;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */


  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex (&yylval, param);
    }

  if (yychar <= YYEOF)
    {
      yychar = YYEOF;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == YYerror)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = YYUNDEF;
      yytoken = YYSYMBOL_YYerror;
      goto yyerrlab1;
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 27: /* dash_com: "(" "dash" floating_point_number ")"  */
#line 161 "player_command_parser.ypp"
           {
             BUILDER.dash( (yyvsp[-1]. m_double ) );
           }
#line 1379 "player_command_parser.cpp"
    break;

  case 28: /* dash_com: "(" "dash" floating_point_number floating_point_number ")"  */
#line 165 "player_command_parser.ypp"
           {
	           BUILDER.dash( (yyvsp[-2]. m_double ), (yyvsp[-1]. m_double ) );
           }
#line 1387 "player_command_parser.cpp"
    break;

  case 34: /* dash_left: "(" RCSS_PCOM_LEFT floating_point_number floating_point_number ")"  */
#line 178 "player_command_parser.ypp"
            {
              BUILDER.dashLeftLeg( (yyvsp[-2]. m_double ), (yyvsp[-1]. m_double ) );
            }
#line 1395 "player_command_parser.cpp"
    break;

  case 35: /* dash_right: "(" RCSS_PCOM_RIGHT floating_point_number floating_point_number ")"  */
#line 184 "player_command_parser.ypp"
             {
               BUILDER.dashRightLeg( (yyvsp[-2]. m_double ), (yyvsp[-1]. m_double ) );
             }
#line 1403 "player_command_parser.cpp"
    break;

  case 36: /* turn_com: "(" "turn" floating_point_number ")"  */
#line 190 "player_command_parser.ypp"
           {
             BUILDER.turn( (yyvsp[-1]. m_double ) );
           }
#line 1411 "player_command_parser.cpp"
    break;

  case 37: /* turn_neck_com: "(" "turn_neck" floating_point_number ")"  */
#line 196 "player_command_parser.ypp"
                {
                  BUILDER.turn_neck( (yyvsp[-1]. m_double ) );
                }
#line 1419 "player_command_parser.cpp"
    break;

  case 38: /* change_focus_com: "(" "change_focus" floating_point_number floating_point_number ")"  */
#line 202 "player_command_parser.ypp"
                {
                  BUILDER.change_focus( (yyvsp[-2]. m_double ), (yyvsp[-1]. m_double ) );
                }
#line 1427 "player_command_parser.cpp"
    break;

  case 39: /* kick_com: "(" "kick" floating_point_number floating_point_number ")"  */
#line 208 "player_command_parser.ypp"
           {
             BUILDER.kick( (yyvsp[-2]. m_double ), (yyvsp[-1]. m_double ) );
           }
#line 1435 "player_command_parser.cpp"
    break;

  case 40: /* long_kick_com: "(" "long_kick" floating_point_number floating_point_number ")"  */
#line 214 "player_command_parser.ypp"
           {
             BUILDER.long_kick( (yyvsp[-2]. m_double ), (yyvsp[-1]. m_double ) );
           }
#line 1443 "player_command_parser.cpp"
    break;

  case 41: /* catch_com: "(" "catch" floating_point_number ")"  */
#line 220 "player_command_parser.ypp"
            {
              BUILDER.goalieCatch( (yyvsp[-1]. m_double ) );
            }
#line 1451 "player_command_parser.cpp"
    break;

  case 42: /* drop_com: "(" "drop" ")"  */
#line 226 "player_command_parser.ypp"
           {
             BUILDER.drop();
           }
#line 1459 "player_command_parser.cpp"
    break;

  case 43: /* say_com: "unquoted say"  */
#line 232 "player_command_parser.ypp"
          {
            BUILDER.say( (yyvsp[0]. m_str ) );
          }
#line 1467 "player_command_parser.cpp"
    break;

  case 44: /* say_com: "(" "say" RCSS_PCOM_STR ")"  */
#line 236 "player_command_parser.ypp"
          {
            BUILDER.say( rcss::stripQuotes( (yyvsp[-1]. m_str ) ) );
          }
#line 1475 "player_command_parser.cpp"
    break;

  case 45: /* sense_body_com: "(" "sense_body" ")"  */
#line 242 "player_command_parser.ypp"
                 {
                   BUILDER.sense_body();
                 }
#line 1483 "player_command_parser.cpp"
    break;

  case 46: /* score_com: "(" "score" ")"  */
#line 248 "player_command_parser.ypp"
            {
              BUILDER.score();
            }
#line 1491 "player_command_parser.cpp"
    break;

  case 47: /* move_com: "(" "move" floating_point_number floating_point_number ")"  */
#line 254 "player_command_parser.ypp"
           {
             BUILDER.move( (yyvsp[-2]. m_double ), (yyvsp[-1]. m_double ) );
           }
#line 1499 "player_command_parser.cpp"
    break;

  case 48: /* change_view_com: "(" "change_view" view_width view_quality ")"  */
#line 260 "player_command_parser.ypp"
                 {
                   BUILDER.change_view( (yyvsp[-2]. m_view_w ), (yyvsp[-1]. m_view_q ) );
                 }
#line 1507 "player_command_parser.cpp"
    break;

  case 49: /* change_view_com: "(" "change_view" view_width ")"  */
#line 264 "player_command_parser.ypp"
                 {
                   BUILDER.change_view( (yyvsp[-1]. m_view_w ) );
                 }
#line 1515 "player_command_parser.cpp"
    break;

  case 50: /* view_width: "narrow"  */
#line 270 "player_command_parser.ypp"
             {
               (yyval. m_view_w ) = rcss::pcom::NARROW;
             }
#line 1523 "player_command_parser.cpp"
    break;

  case 51: /* view_width: "normal"  */
#line 274 "player_command_parser.ypp"
             {
               (yyval. m_view_w ) = rcss::pcom::NORMAL;
             }
#line 1531 "player_command_parser.cpp"
    break;

  case 52: /* view_width: "wide"  */
#line 278 "player_command_parser.ypp"
             {
               (yyval. m_view_w ) = rcss::pcom::WIDE;
             }
#line 1539 "player_command_parser.cpp"
    break;

  case 53: /* view_quality: "low"  */
#line 284 "player_command_parser.ypp"
               {
                 (yyval. m_view_q ) = rcss::pcom::LOW;
               }
#line 1547 "player_command_parser.cpp"
    break;

  case 54: /* view_quality: "high"  */
#line 288 "player_command_parser.ypp"
               {
                 (yyval. m_view_q ) = rcss::pcom::HIGH;
               }
#line 1555 "player_command_parser.cpp"
    break;

  case 55: /* compression_com: "(" "compression" RCSS_PCOM_INT ")"  */
#line 294 "player_command_parser.ypp"
                  {
                    BUILDER.compression( (yyvsp[-1]. m_int ) );
                  }
#line 1563 "player_command_parser.cpp"
    break;

  case 56: /* bye_com: "(" "bye" ")"  */
#line 300 "player_command_parser.ypp"
          {
            BUILDER.bye();
          }
#line 1571 "player_command_parser.cpp"
    break;

  case 57: /* done_com: "(" "done" ")"  */
#line 306 "player_command_parser.ypp"
           {
             BUILDER.done();
           }
#line 1579 "player_command_parser.cpp"
    break;

  case 58: /* pointto_com: "(" "pointto" floating_point_number floating_point_number ")"  */
#line 312 "player_command_parser.ypp"
              {
                BUILDER.pointto( true, (yyvsp[-2]. m_double ), (yyvsp[-1]. m_double ) );
              }
#line 1587 "player_command_parser.cpp"
    break;

  case 59: /* pointto_com: "(" "pointto" "off" ")"  */
#line 316 "player_command_parser.ypp"
              {
                BUILDER.pointto( false, 0.0, 0.0 );
              }
#line 1595 "player_command_parser.cpp"
    break;

  case 60: /* attentionto_com: "(" "attentionto" team_side RCSS_PCOM_INT ")"  */
#line 322 "player_command_parser.ypp"
                  {
                    BUILDER.attentionto( true, (yyvsp[-2]. m_team ), "", (yyvsp[-1]. m_int ) );
                  }
#line 1603 "player_command_parser.cpp"
    break;

  case 61: /* attentionto_com: "(" "attentionto" RCSS_PCOM_STR RCSS_PCOM_INT ")"  */
#line 326 "player_command_parser.ypp"
                  {
                    BUILDER.attentionto( true, rcss::pcom::UNKNOWN_TEAM, (yyvsp[-2]. m_str ), (yyvsp[-1]. m_int ) );
                  }
#line 1611 "player_command_parser.cpp"
    break;

  case 62: /* attentionto_com: "(" "attentionto" "off" ")"  */
#line 330 "player_command_parser.ypp"
                  {
                    BUILDER.attentionto( false, rcss::pcom::UNKNOWN_TEAM, "", 0 );
                  }
#line 1619 "player_command_parser.cpp"
    break;

  case 63: /* tackle_com: "(" "tackle" floating_point_number ")"  */
#line 336 "player_command_parser.ypp"
             {
               BUILDER.tackle( (yyvsp[-1]. m_double ) );
             }
#line 1627 "player_command_parser.cpp"
    break;

  case 64: /* tackle_com: "(" "tackle" floating_point_number boolean_value ")"  */
#line 341 "player_command_parser.ypp"
             {
               BUILDER.tackle( (yyvsp[-2]. m_double ), (yyvsp[-1]. m_bool ) );
             }
#line 1635 "player_command_parser.cpp"
    break;

  case 65: /* clang_com: "(" "clang" "(" "ver" RCSS_PCOM_INT RCSS_PCOM_INT ")" ")"  */
#line 347 "player_command_parser.ypp"
           {
             BUILDER.clang( (yyvsp[-3]. m_int ), (yyvsp[-2]. m_int ) );
           }
#line 1643 "player_command_parser.cpp"
    break;

  case 66: /* ear_com: "(" "ear" "(" on_off team_side partial_complete ")" ")"  */
#line 353 "player_command_parser.ypp"
          {
            BUILDER.ear( (yyvsp[-4]. m_bool ), (yyvsp[-3]. m_team ), "", (yyvsp[-2]. m_ear ) );
          }
#line 1651 "player_command_parser.cpp"
    break;

  case 67: /* ear_com: "(" "ear" "(" on_off RCSS_PCOM_STR partial_complete ")" ")"  */
#line 357 "player_command_parser.ypp"
          {
            BUILDER.ear( (yyvsp[-4]. m_bool ), rcss::pcom::UNKNOWN_TEAM, (yyvsp[-3]. m_str ), (yyvsp[-2]. m_ear ) );
          }
#line 1659 "player_command_parser.cpp"
    break;

  case 68: /* ear_com: "(" "ear" "(" on_off team_side ")" ")"  */
#line 361 "player_command_parser.ypp"
          {
            BUILDER.ear( (yyvsp[-3]. m_bool ), (yyvsp[-2]. m_team ), "", rcss::pcom::UNKNOWN_EAR_MODE );
          }
#line 1667 "player_command_parser.cpp"
    break;

  case 69: /* ear_com: "(" "ear" "(" on_off RCSS_PCOM_STR ")" ")"  */
#line 365 "player_command_parser.ypp"
          {
            BUILDER.ear( (yyvsp[-3]. m_bool ), rcss::pcom::UNKNOWN_TEAM, (yyvsp[-2]. m_str ), rcss::pcom::UNKNOWN_EAR_MODE );
          }
#line 1675 "player_command_parser.cpp"
    break;

  case 70: /* ear_com: "(" "ear" "(" on_off partial_complete ")" ")"  */
#line 369 "player_command_parser.ypp"
          {
            BUILDER.ear( (yyvsp[-3]. m_bool ), rcss::pcom::UNKNOWN_TEAM, "", (yyvsp[-2]. m_ear ) );
          }
#line 1683 "player_command_parser.cpp"
    break;

  case 71: /* ear_com: "(" "ear" "(" on_off ")" ")"  */
#line 373 "player_command_parser.ypp"
          {
            BUILDER.ear( (yyvsp[-2]. m_bool ), rcss::pcom::UNKNOWN_TEAM, "", rcss::pcom::UNKNOWN_EAR_MODE );
          }
#line 1691 "player_command_parser.cpp"
    break;

  case 72: /* synch_see_com: "(" "synch_see" ")"  */
#line 379 "player_command_parser.ypp"
                {
                   BUILDER.synch_see();
                }
#line 1699 "player_command_parser.cpp"
    break;

  case 73: /* gaussian_see_com: "(" "gaussian_see" ")"  */
#line 385 "player_command_parser.ypp"
                {
                   BUILDER.gaussian_see();
                }
#line 1707 "player_command_parser.cpp"
    break;

  case 74: /* on_off: "on"  */
#line 391 "player_command_parser.ypp"
         {
           (yyval. m_bool ) = true;
         }
#line 1715 "player_command_parser.cpp"
    break;

  case 75: /* on_off: "off"  */
#line 395 "player_command_parser.ypp"
         {
           (yyval. m_bool ) = false;
         }
#line 1723 "player_command_parser.cpp"
    break;

  case 76: /* boolean_value: "on"  */
#line 401 "player_command_parser.ypp"
                {
                  (yyval. m_bool ) = true;
                }
#line 1731 "player_command_parser.cpp"
    break;

  case 77: /* boolean_value: "off"  */
#line 405 "player_command_parser.ypp"
                {
                  (yyval. m_bool ) = false;
                }
#line 1739 "player_command_parser.cpp"
    break;

  case 78: /* boolean_value: "true"  */
#line 409 "player_command_parser.ypp"
                {
                  (yyval. m_bool ) = true;
                }
#line 1747 "player_command_parser.cpp"
    break;

  case 79: /* boolean_value: "false"  */
#line 413 "player_command_parser.ypp"
                {
                  (yyval. m_bool ) = false;
                }
#line 1755 "player_command_parser.cpp"
    break;

  case 80: /* team_side: "our"  */
#line 419 "player_command_parser.ypp"
            {
              (yyval. m_team ) = rcss::pcom::OUR;
            }
#line 1763 "player_command_parser.cpp"
    break;

  case 81: /* team_side: "opp"  */
#line 423 "player_command_parser.ypp"
            {
              (yyval. m_team ) = rcss::pcom::OPP;
            }
#line 1771 "player_command_parser.cpp"
    break;

  case 82: /* team_side: RCSS_PCOM_LEFT  */
#line 427 "player_command_parser.ypp"
            {
              (yyval. m_team ) = rcss::pcom::LEFT_SIDE;
            }
#line 1779 "player_command_parser.cpp"
    break;

  case 83: /* team_side: RCSS_PCOM_RIGHT  */
#line 431 "player_command_parser.ypp"
            {
              (yyval. m_team ) = rcss::pcom::RIGHT_SIDE;
            }
#line 1787 "player_command_parser.cpp"
    break;

  case 84: /* partial_complete: "partial"  */
#line 437 "player_command_parser.ypp"
                   {
                     (yyval. m_ear ) = rcss::pcom::PARTIAL;
                   }
#line 1795 "player_command_parser.cpp"
    break;

  case 85: /* partial_complete: "complete"  */
#line 441 "player_command_parser.ypp"
                   {
                     (yyval. m_ear ) = rcss::pcom::COMPLETE;
                   }
#line 1803 "player_command_parser.cpp"
    break;

  case 86: /* floating_point_number: RCSS_PCOM_INT  */
#line 447 "player_command_parser.ypp"
                        {
                          (yyval. m_double ) = static_cast< double >( (yyvsp[0]. m_int ) );
                        }
#line 1811 "player_command_parser.cpp"
    break;

  case 87: /* floating_point_number: RCSS_PCOM_REAL  */
#line 451 "player_command_parser.ypp"
                        {
                          (yyval. m_double ) = (yyvsp[0]. m_double );
                        }
#line 1819 "player_command_parser.cpp"
    break;


#line 1823 "player_command_parser.cpp"

      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      yyerror (param, YY_("syntax error"));
    }

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval, param);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;
  ++yynerrs;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp, param);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturnlab;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturnlab;


/*-----------------------------------------------------------.
| yyexhaustedlab -- YYNOMEM (memory exhaustion) comes here.  |
`-----------------------------------------------------------*/
yyexhaustedlab:
  yyerror (param, YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturnlab;


/*----------------------------------------------------------.
| yyreturnlab -- parsing is finished, clean up and return.  |
`----------------------------------------------------------*/
yyreturnlab:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval, param);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp, param);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif

  return yyresult;
}

#line 456 "player_command_parser.ypp"



void yyerror (rcss::pcom::Parser::Param& /*param*/, const char* s)
{
  std::cerr << s << std::endl;
  //do nothing
}

int yyerror (rcss::pcom::Parser::Param& param, char* s)
{
  yyerror ( param, (const char*)s );
  return 0;
}
