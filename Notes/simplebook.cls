\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{simplebook}[2024/08/11 Simple Book Class]

% Load the base class
\LoadClass{ctexbook}

% Set up essential packages
\RequirePackage{amsmath, amssymb, amsthm, amsfonts}  % Math packages
\RequirePackage{graphicx}  % Including graphics
\RequirePackage[colorlinks]{hyperref}  % Hyperlinks in PDF
\RequirePackage{titlesec}  % Custom section titles
\RequirePackage{fancyhdr}  % Custom headers and footers
\RequirePackage{color}     % Color support
\RequirePackage{babel}     % Language support
\RequirePackage{lmodern}   % Modern fonts
\RequirePackage{romannum}  % Roman numerals support
\RequirePackage{ccicons}   % Creative Commons licenses
\RequirePackage{braket}    % Inner product notation
\RequirePackage{float}     % Improved floating environments
\RequirePackage{framed}    % Framed boxes
\RequirePackage{tikz}      % TikZ for drawing
\RequirePackage{subcaption}    % Subfigures
\RequirePackage{tocloft}   % Customize TOC appearance

% Set page layout
\RequirePackage{geometry}
\geometry{
  a4paper,
  top=25.4mm, bottom=25.4mm,
  left=20mm, right=20mm,
  headheight=2.17cm,
  headsep=4mm,
  footskip=12mm
}

% Define theorem environments
\theoremstyle{plain}
\newtheorem{theorem}{定理}[chapter]
\newtheorem{lemma}{引理}[chapter]
\newtheorem{proposition}{命题}[chapter]
\newtheorem*{corollary}{推论}

\theoremstyle{definition}
\newtheorem{definition}{定义}[chapter]
\newtheorem{conjecture}{猜想}[chapter]
\newtheorem{example}{例}[chapter]

\theoremstyle{remark}
\newtheorem*{remark}{\normalfont\bfseries 评论}
\newtheorem*{note}{\normalfont\bfseries 注}
\newtheorem{case}{\normalfont\bfseries 案例}

\renewcommand*{\proofname}{\normalfont\bfseries 证明}

% Customize document names
\renewcommand\contentsname{目录}
\renewcommand\bibname{参考文献}
\renewcommand\figurename{图}
\renewcommand\tablename{表}
\renewcommand\abstractname{摘\hspace{2em}要}

% Additional custom commands
\newcommand{\keywords}[1]{\vskip2ex\par\noindent\normalfont{\bfseries 关键词: } #1}
\newcommand\figref[1]{{\bfseries 图~\ref{#1}}}
\newcommand\tabref[1]{{\bfseries 表~\ref{#1}}}
%\setlength\parindent{2\ccwd}
\renewcommand{\theequation}{\thesection.\arabic{equation}}

% Customize TOC appearance
\renewcommand{\cftchapfont}{\bfseries}
\renewcommand{\cftchappagefont}{\bfseries}
%\setcounter{secnumdepth}{4}  % Enable four-level numbering

% Set default figure/table placement parameters
\setcounter{topnumber}{2}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}
\renewcommand{\topfraction}{0.85}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.15}

% Title page setup
\renewcommand{\maketitle}{
  \begin{titlepage}
    \centering
    \vspace*{3cm}
    {\Huge \@title \par}
    \vspace{1cm}
    {\Large \@author \par}
    \vfill
    {\Large \@date\par}
    \vspace*{3cm}
  \end{titlepage}
  \clearpage
  \thispagestyle{empty}  % Remove page number on the first page after title
  \clearpage
}

% End of class
\endinput

