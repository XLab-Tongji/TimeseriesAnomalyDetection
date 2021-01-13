# High-level Design Document

[toc]

## Architecture Design

When considering the product architecture, a detailed three-level hierarchy is adopted, which divides the entire business into: Presentation layer, Business Logic Layer and Data access layer. The idea of "divide and conquer" is adopted to divide the problem into individual solutions, which is easy to control, extend and allocate resources.

![architect](img\architect.png)

### Presentation Layer

Located at the outermost level (top level), closest to the user. Used to display data and receive user input data. Provide an interactive operation interface for users. Mainly accept user requests and return data. Provide client access to applications.

### Business Logic Layer

Business Logic Layer (Business Logic Layer) is the core value part of the system architecture. Mainly responsible for the operation of the data layer, that is to say the combination of some data layer operations. It is located between the service layer and the data access layer, and plays a role in data exchange.

### Data Access Layer

It is mainly the operation layer for non-raw data (database or text file storage form), rather than the original data, that is to say, it is the operation of the database, not the data. It provides data services for the business logic layer or presentation layer, such as adding, deleting, modifying, and searching for data.

## Interface Design